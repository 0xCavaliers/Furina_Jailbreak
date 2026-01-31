import torch
import json
import numpy as np
import os
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Import from refusal_direction project
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import get_mean_activations
from pipeline.utils.hook_utils import add_hooks
from dataset.load_dataset import load_dataset_split

def calculate_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Calculate cosine similarity between two vectors"""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).item()

def load_advbench_entropy_datasets(dataset_path: str, entropy_levels: List[str] = ['low', 'medium', 'high', 'semantic'],
                                  max_samples_per_level: int = 50) -> Dict[str, List[str]]:
    """
    Load different entropy level prompt datasets from advbench_harmful_questions.json

    Args:
        dataset_path: dataset file path
        entropy_levels: entropy level list
        max_samples_per_level: maximum number of samples per level

    Returns:
        Dictionary of lists of prompts for each entropy level
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entropy_datasets = {level: [] for level in entropy_levels}

    # Define entropy level mapping
    entropy_mapping = {
        'low': ['original_question', 'minor_rewrite_question'],
        'medium': ['moderate_rewrite_question'],
        'high': ['high_rewrite_question'],
        'semantic': ['semantic_rewrite_question']
    }

    # Collect all samples
    for item_id, item_data in data.items():
        for entropy_level, question_types in entropy_mapping.items():
            if entropy_level in entropy_levels:
                for q_type in question_types:
                    if q_type in item_data:
                        entropy_datasets[entropy_level].append(item_data[q_type])

    # Limit sample number and random sample
    for level in entropy_levels:
        if len(entropy_datasets[level]) > max_samples_per_level:
            entropy_datasets[level] = random.sample(entropy_datasets[level], max_samples_per_level)

    # Print statistics
    print("Dataset loaded:")
    for level, prompts in entropy_datasets.items():
        print(f"  {level}: {len(prompts)} samples")

    return entropy_datasets

def extract_activations_for_entropy_levels(
    model_base,
    entropy_datasets: Dict[str, List[str]],
    target_layer: int,
    batch_size: int = 8
) -> Dict[str, torch.Tensor]:
    """
    Extract activations for different entropy level prompts at the specified layer

    Args:
        model_base: model base class
        entropy_datasets: different entropy level prompt datasets
        target_layer: target layer index
        batch_size: batch size

    Returns:
        Dictionary of activation tensors for each entropy level
    """
    activations = {}

    for entropy_level, prompts in entropy_datasets.items():
        if not prompts:  # skip empty dataset
            continue

        print(f"Extracting activations for {entropy_level} entropy prompts...")

        # Create activation cache
        n_samples = len(prompts)
        d_model = model_base.model.config.hidden_size
        activation_cache = torch.zeros((1, model_base.model.config.num_hidden_layers, d_model),
                                     dtype=torch.float32, device=model_base.model.device)

        def activation_hook(module, input, output=None):
            """Hook function to capture activations"""
            activation = input[0].clone().detach()
            # Only take activations of the last token
            activation_cache[:, module.layer_idx, :] += activation[:, -1, :].mean(dim=0, keepdim=True)

        # Add hook to target layer
        hooks = []
        for i, block_module in enumerate(model_base.model_block_modules):
            block_module.layer_idx = i  # Add layer index
            if i == target_layer:
                hook = block_module.register_forward_hook(activation_hook)
                hooks.append(hook)

        # Batch process
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            inputs = model_base.tokenize_instructions_fn(instructions=batch_prompts)

            with torch.no_grad():
                model_base.model(
                    input_ids=inputs.input_ids.to(model_base.model.device),
                    attention_mask=inputs.attention_mask.to(model_base.model.device)
                )

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Store activations
        activations[entropy_level] = activation_cache[0, target_layer, :].cpu()

    return activations

def analyze_entropy_vs_refusal_direction(
    model_paths: List[str],
    dataset_path: str,
    entropy_levels: List[str] = ['low', 'medium', 'high', 'semantic'],
    target_layer: int = 14,
    output_dir: str = './entropy_analysis_results',
    max_samples_per_level: int = 50
) -> Dict:
    """
    Analyze the similarity between different entropy level prompts and refusal direction

    Args:
        model_paths: model path list
        dataset_path: dataset path
        entropy_levels: entropy level list
        target_layer: target layer
        output_dir: output directory
        max_samples_per_level: maximum number of samples per level

    Returns:
        Analysis result dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load entropy dataset
    entropy_datasets = load_advbench_entropy_datasets(
        dataset_path, entropy_levels, max_samples_per_level
    )

    results = {}

    for model_path in model_paths:
        print(f"\n=== Analyzing model: {os.path.basename(model_path)} ===")

        try:
            # Construct model
            model_base = construct_model_base(model_path)
            model_name = os.path.basename(model_path)

            # Check if pre-computed refusal direction exists
            direction_path = f'pipeline/runs/{model_name}/direction.pt'
            if os.path.exists(direction_path):
                refusal_direction = torch.load(direction_path, map_location='cpu')
                print(f"Loaded pre-computed refusal direction from {direction_path}")
            else:
                print(f"No pre-computed direction found for {model_name}, skipping...")
                continue

            # Extract activations for each entropy level
            activations = extract_activations_for_entropy_levels(
                model_base, entropy_datasets, target_layer
            )

            # Calculate similarity
            similarities = {}
            for entropy_level, activation in activations.items():
                similarity = calculate_cosine_similarity(
                    activation.to(refusal_direction.device),
                    refusal_direction
                )
                similarities[entropy_level] = similarity
                print(f"{entropy_level:<10} similarity: {similarity:.4f}")

            results[model_name] = {
                'similarities': similarities,
                'activations': {k: v.tolist() for k, v in activations.items()},
                'refusal_direction': refusal_direction.tolist(),
                'target_layer': target_layer
            }

        except Exception as e:
            print(f"Error processing model {model_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    with open(f'{output_dir}/entropy_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate comparison chart
    plot_entropy_comparison(results, output_dir)

    return results

def plot_entropy_comparison(results: Dict, output_dir: str):
    """Generate entropy level comparison chart"""
    if not results:
        return

    entropy_levels = list(next(iter(results.values()))['similarities'].keys())
    model_names = list(results.keys())

    # Prepare data
    data = {}
    for level in entropy_levels:
        data[level] = [results[model]['similarities'].get(level, 0) for model in model_names]

    # Create chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(entropy_levels))
    width = 0.35

    for i, model in enumerate(model_names):
        values = [data[level][i] for level in entropy_levels]
        ax.bar(x + i*width, values, width, label=model, alpha=0.8)

    ax.set_xlabel('Entropy Level')
    ax.set_ylabel('Cosine Similarity with Refusal Direction')
    ax.set_title('Entropy Level vs Refusal Direction Similarity (AdvBench Dataset)')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([level.capitalize() for level in entropy_levels])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add numerical labels
    for i, model in enumerate(model_names):
        values = [data[level][i] for level in entropy_levels]
        for j, v in enumerate(values):
            ax.text(j + i*width, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/entropy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_summary(results: Dict):
    print("\n" + "="*80)
    print("ENTROPY vs REFUSAL DIRECTION ANALYSIS SUMMARY (AdvBench Dataset)")
    print("="*80)

    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        print("-" * 50)

        similarities = model_results['similarities']
        for entropy_level, similarity in similarities.items():
            print(f"  {entropy_level:<15}: {similarity:.4f}")

        # Difference analysis
        if 'low' in similarities and 'semantic' in similarities:
            diff = similarities['semantic'] - similarities['low']
            print(f"\n  Δ(semantic - low): {diff:+.4f}")

        # Trend analysis
        ordered_levels = ['low', 'medium', 'high', 'semantic']
        values = [similarities[l] for l in ordered_levels if l in similarities]

        if len(values) >= 2:
            trend = "increasing" if values[0] < values[-1] else "decreasing"
            print(f"  Trend: {trend} alignment with higher entropy")

def main():
    parser = argparse.ArgumentParser(description="Analyze entropy levels vs refusal direction similarity using AdvBench dataset")
    parser.add_argument('--models', nargs='+',
                       default=['/root/shared-nvme/LLMs/Llama-2-7b-chat-hf',
                               '/root/shared-nvme/LLMs/Qwen3-8B'],
                       help='Model paths to analyze')
    parser.add_argument('--dataset_path', type=str,
                       default='/root/shared-nvme/advbench_harmful_questions.json',
                       help='Path to AdvBench harmful questions dataset')
    parser.add_argument('--entropy_levels', nargs='+',
                       default=['low', 'medium', 'high', 'semantic'],
                       help='Entropy levels to analyze')
    parser.add_argument('--target_layer', type=int, default=14,
                       help='Target layer for activation extraction')
    parser.add_argument('--output_dir', type=str,
                       default='./entropy_analysis_results',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=50,
                       help='Maximum samples per entropy level')

    args = parser.parse_args()

    # Run analysis
    results = analyze_entropy_vs_refusal_direction(
        model_paths=args.models,
        dataset_path=args.dataset_path,
        entropy_levels=args.entropy_levels,
        target_layer=args.target_layer,
        output_dir=args.output_dir,
        max_samples_per_level=args.max_samples
    )

    # Print summary
    print_summary(results)

    print(f"\nResults saved to {args.output_dir}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
