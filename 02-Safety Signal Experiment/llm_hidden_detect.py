import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional, Union
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# List of Refusal Words - Based on the HiddenDetect Project
REFUSAL_TOKENS = [
    "Sorry", "sorry",
    "unfortunately", "unfortunate", "sadly",
    "explicit", "deadly", "crim", "criminal", "illegal", "dangerous", "harmful",
    "warning", "alarm", "caution", "shame", "conspiracy",
    "Subject", "contrary", "shouldn't", "cannot", "unable", "decline", "refuse"
]

class LLMAdapter:
    """Generic LLM Adapter Base Class"""
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.refusal_token_ids = []
        self.vocab_size = 0

    def load_model(self):
        """Load model and tokenizer"""
        raise NotImplementedError

    def preprocess_input(self, text: str) -> torch.Tensor:
        """Preprocess input text"""
        raise NotImplementedError

    def get_hidden_states(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Get hidden states"""
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states

    def get_normalized_output(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Get normalized output"""
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            norm = self.model.model.norm
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            norm = self.model.transformer.ln_f
        else:
            # Default use LayerNorm
            norm = torch.nn.LayerNorm(hidden_state.shape[-1], device=hidden_state.device)
        return norm(hidden_state)

    def get_next_token_logits(self, normalized_output: torch.Tensor) -> torch.Tensor:
        """Get next token logits"""
        return self.model.lm_head(normalized_output)[:, -1, :]

    def prepare_refusal_vector(self):
        """Prepare refusal vector"""
        if not self.refusal_token_ids:
            for token in REFUSAL_TOKENS:
                try:
                    token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
                    self.refusal_token_ids.append(token_id)
                except:
                    continue

        self.refusal_vector = torch.zeros(self.vocab_size)
        for token_id in self.refusal_token_ids:
            if token_id < self.vocab_size:
                self.refusal_vector[token_id] = 1.0

    def cleanup(self):
        """Clean up memory"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("Memory cleaned up")

class LlamaAdapter(LLMAdapter):
    """Llama Model Adapter"""

    def __init__(self, model_path: str, device: str = "auto"):
        super().__init__(model_path, device)
        self.load_model()

    def load_model(self):
        """Load Llama model"""
        print(f"Loading Llama model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # Add pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {
            "device_map": self.device,
            "torch_dtype": torch.float16,
            "trust_remote_code": True
        }

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        self.vocab_size = self.model.config.vocab_size
        self.prepare_refusal_vector()
        print("Llama model loaded successfully!")

    def preprocess_input(self, text: str) -> torch.Tensor:
        """Preprocess Llama input"""
        # Use simple chat format
        chat_template = f"<s>[INST] {text} [/INST]"
        input_ids = self.tokenizer.encode(chat_template, return_tensors="pt")
        return input_ids.to(self.model.device)

class QwenAdapter(LLMAdapter):
        """Qwen Model Adapter"""

    def __init__(self, model_path: str, device: str = "auto"):
        super().__init__(model_path, device)
        self.load_model()

    def load_model(self):
        """Load Qwen model"""
        print(f"Loading Qwen model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.vocab_size = self.model.config.vocab_size
        self.prepare_refusal_vector()
        print("Qwen model loaded successfully!")

    def preprocess_input(self, text: str) -> torch.Tensor:
        """Preprocess Qwen input"""
        # Use Qwen's chatml format
        messages = [{"role": "user", "content": text}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        return input_ids.to(self.model.device)

class HiddenDetectLLM:
    """LLM Hidden Detector - Calculate Refusal Discrepancy Trajectory"""

    def __init__(self, model_path: str, model_type: str = "llama"):
        """
        Initialize detector

        Args:
            model_path: model path
            model_type: model type ("llama" or "qwen")
        """
        self.model_path = model_path
        self.model_type = model_type.lower()

        if self.model_type == "llama":
            self.adapter = LlamaAdapter(model_path)
        elif self.model_type == "qwen":
            self.adapter = QwenAdapter(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Set detection layer range (can be adjusted based on the model)
        self.layer_configs = {
            "llama": {"start": 16, "end": 29},  # Llama-2-7B
            "qwen": {"start": 21, "end": 24}    # Qwen-7B
        }

    def extract_refusal_curve(self, text: str, layer_range: Optional[tuple] = None) -> Dict[int, float]:
        """
        Extract layer-wise refusal curve for a single text (corresponds to Figure 3 in the paper)

        Args:
            text: input text
            layer_range: detection layer range (start, end), if None then use default value

        Returns:
            Dictionary {layer_idx: refusal_score}, refusal_score is the cosine similarity of this layer
        """
        if layer_range is None:
            config = self.layer_configs[self.model_type]
            start_layer, end_layer = config["start"], config["end"]
        else:
            start_layer, end_layer = layer_range

        # Preprocess input
        input_ids = self.adapter.preprocess_input(text)

        # Get hidden states
        hidden_states = self.adapter.get_hidden_states(input_ids)

        # Calculate refusal score for each layer
        refusal_curve = {}
        for layer_idx in range(start_layer, end_layer + 1):
            if layer_idx >= len(hidden_states):
                continue

            hidden_state = hidden_states[layer_idx]
            normalized_output = self.adapter.get_normalized_output(hidden_state)
            next_token_logits = self.adapter.get_next_token_logits(normalized_output)

            # Calculate cosine similarity with refusal vector
            refusal_logits = self.adapter.refusal_vector.to(next_token_logits.device)
            cos_sim = F.cosine_similarity(next_token_logits, refusal_logits, dim=-1)
            refusal_curve[layer_idx] = cos_sim.item()

        return refusal_curve

    def aggregate_refusal_curves(self, dataset: List[Dict], layer_range: Optional[tuple] = None) -> tuple:
        """
        Aggregate refusal curves for a dataset, calculate average values for safe/unsafe groups

        Args:
            dataset: dataset, each sample contains 'text' and 'toxicity' fields
            layer_range: detection layer range

        Returns:
            (safe_curves_dict, unsafe_curves_dict)
            where each dict format is {layer_idx: mean_refusal_score}
        """
        safe_curves = []
        unsafe_curves = []

        print(f"Processing {len(dataset)} samples for refusal curves...")
        for i, sample in enumerate(dataset):
            if i % 50 == 0:
                print(f"Processing sample {i+1}/{len(dataset)}")

            text = sample.get('text', sample.get('txt', ''))
            toxicity = sample.get('toxicity', sample.get('label', 0))

            # Process label format
            if isinstance(toxicity, str):
                toxicity = 1 if toxicity.lower() in ['unsafe', 'toxic', '1'] else 0

            # Extract refusal curve
            curve = self.extract_refusal_curve(text, layer_range)

            if toxicity == 0:
                safe_curves.append(curve)
            else:
                unsafe_curves.append(curve)

        # Calculate average curve
        def compute_mean_curve(curves_list):
            if not curves_list:
                return {}
            # Get all layers
            all_layers = set()
            for curve in curves_list:
                all_layers.update(curve.keys())

            mean_curve = {}
            for layer in sorted(all_layers):
                layer_scores = [curve.get(layer, 0) for curve in curves_list if layer in curve]
                if layer_scores:
                    mean_curve[layer] = np.mean(layer_scores)

            return mean_curve

        safe_mean_curve = compute_mean_curve(safe_curves)
        unsafe_mean_curve = compute_mean_curve(unsafe_curves)

        print(f"Safe samples: {len(safe_curves)}, Unsafe samples: {len(unsafe_curves)}")
        print(f"Safe curve layers: {len(safe_mean_curve)}, Unsafe curve layers: {len(unsafe_mean_curve)}")

        return safe_mean_curve, unsafe_mean_curve

    def compute_refusal_discrepancy(self, safe_curve: Dict[int, float],
                                   unsafe_curve: Dict[int, float]) -> Dict[int, float]:
        """
        Calculate Refusal Discrepancy: unsafe_mean - safe_mean (corresponds to Figure 3 in the paper)

        Args:
            safe_curve: average refusal curve for safe samples {layer_idx: mean_score}
            unsafe_curve: average refusal curve for unsafe samples {layer_idx: mean_score}

        Returns:
            discrepancy curve {layer_idx: discrepancy}
        """
        # Get union of all layers
        all_layers = set(safe_curve.keys()) | set(unsafe_curve.keys())

        discrepancy_curve = {}
        for layer in sorted(all_layers):
            safe_score = safe_curve.get(layer, 0)
            unsafe_score = unsafe_curve.get(layer, 0)
            discrepancy_curve[layer] = unsafe_score - safe_score

        return discrepancy_curve

def load_custom_dataset(file_path: str, format: str = "auto") -> List[Dict]:
    """
    Load custom dataset

    Args:
        file_path: file path
        format: file format ("json", "csv", "auto")

    Returns:
        dataset list
    """
    dataset = []

    if format == "auto":
        if file_path.endswith('.json'):
            format = "json"
        elif file_path.endswith('.csv'):
            format = "csv"
        else:
            raise ValueError("Cannot auto-detect format. Please specify format parameter.")

    if format == "json":
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            sample = {
                "text": item.get("text", item.get("prompt", item.get("txt", ""))),
                "toxicity": item.get("toxicity", item.get("label", item.get("is_safe", True))) ^ 1
                if isinstance(item.get("is_safe"), bool) else
                item.get("toxicity", item.get("label", 0))
            }
            dataset.append(sample)

    elif format == "csv":
        df = pd.read_csv(file_path)

        # Check column names and handle accordingly
        if "question" in df.columns and "FigTxt" in df.columns:
            # benign_questions.csv format - safe samples
            for _, row in df.iterrows():
                sample = {
                    "text": str(row["question"]),
                    "toxicity": int(row["FigTxt"]) if str(row["FigTxt"]).isdigit() else 0
                }
                dataset.append(sample)
        elif "instruction" in df.columns:
            # safebench.csv格式 - 不安全样本
            for _, row in df.iterrows():
                sample = {
                    "text": str(row["instruction"]),
                    "toxicity": 1  # safebench are all unsafe samples
                }
                dataset.append(sample)
        else:
            # Generic CSV format
            for _, row in df.iterrows():
                sample = {
                    "text": str(row.get("text", row.get("prompt", row.get("txt", "")))),
                    "toxicity": 1 if str(row.get("toxicity", row.get("label", "0"))).lower() in
                               ['unsafe', 'toxic', '1', 'true'] else 0
                }
                dataset.append(sample)

    print(f"Loaded {len(dataset)} samples from {file_path}")
    return dataset

def load_balanced_dataset(safe_path: str, unsafe_path: str, max_samples: int = None,
                         safe_format: str = "csv", unsafe_format: str = "csv") -> List[Dict]:
    """
    Load balanced dataset (safe samples + unsafe samples)

    Args:
        safe_path: safe samples file path
        unsafe_path: unsafe samples file path
        max_samples: maximum number of samples per category (None means use all)
        safe_format: safe samples file format
        unsafe_format: unsafe samples file format

    Returns:
        balanced dataset list
    """
    # Load original dataset
    safe_dataset = load_custom_dataset(safe_path, safe_format)
    unsafe_dataset = load_custom_dataset(unsafe_path, unsafe_format)

    # Filter and limit sample数量
    safe_samples = []
    unsafe_samples = []

    # Process safe samples
    for sample in safe_dataset:
        if sample.get("toxicity", 0) == 0:  # Ensure safe sample
            safe_samples.append(sample)
            if max_samples and len(safe_samples) >= max_samples:
                break

    # Process unsafe samples
    for sample in unsafe_dataset:
        if sample.get("toxicity", 0) == 1:  # Ensure unsafe sample
            unsafe_samples.append(sample)
            if max_samples and len(unsafe_samples) >= max_samples:
                break

    print(f"Loaded: {len(safe_samples)} safe samples, {len(unsafe_samples)} unsafe samples")

    # Use smaller category size to balance dataset
    min_samples = min(len(safe_samples), len(unsafe_samples))
    if min_samples == 0:
        raise ValueError("No samples found in one or both datasets")

    if max_samples:
        min_samples = min(min_samples, max_samples)

    # Balance sampling
    safe_samples = safe_samples[:min_samples]
    unsafe_samples = unsafe_samples[:min_samples]

    # Merge dataset
    balanced_dataset = safe_samples + unsafe_samples

    # Random shuffle
    import random
    random.shuffle(balanced_dataset)

    print(f"Created balanced dataset: {len(safe_samples)} safe + {len(unsafe_samples)} unsafe = {len(balanced_dataset)} total")

    return balanced_dataset

def analyze_dataset_balance(dataset: List[Dict]) -> dict:
    """
    Analyze dataset balance

    Args:
        dataset: dataset

    Returns:
        statistics dictionary
    """
    safe_count = sum(1 for s in dataset if s["toxicity"] == 0)
    unsafe_count = sum(1 for s in dataset if s["toxicity"] == 1)
    total = len(dataset)

    return {
        "total_samples": total,
        "safe_samples": safe_count,
        "unsafe_samples": unsafe_count,
        "safe_percentage": safe_count / total * 100 if total > 0 else 0,
        "unsafe_percentage": unsafe_count / total * 100 if total > 0 else 0,
        "balance_ratio": min(safe_count, unsafe_count) / max(safe_count, unsafe_count) if max(safe_count, unsafe_count) > 0 else 0
    }

def plot_refusal_trajectory(safe_curve: Dict[int, float], unsafe_curve: Dict[int, float],
                          discrepancy_curve: Dict[int, float], title: str = "Refusal Discrepancy Trajectory",
                          filename: str = "refusal_trajectory.png"):
    """
    Plot refusal trajectory (corresponds to Figure 3 in the paper)

    Args:
        safe_curve: average refusal curve for safe samples
        unsafe_curve: average refusal curve for unsafe samples
        discrepancy_curve: discrepancy curve
        title: chart title
        filename: saved file name
    """
    try:
        import matplotlib.pyplot as plt

        # Get all layers
        all_layers = sorted(set(safe_curve.keys()) | set(unsafe_curve.keys()) | set(discrepancy_curve.keys()))

        # Prepare data
        layers = []
        safe_scores = []
        unsafe_scores = []
        discrepancies = []

        for layer in all_layers:
            layers.append(layer)
            safe_scores.append(safe_curve.get(layer, 0))
            unsafe_scores.append(unsafe_curve.get(layer, 0))
            discrepancies.append(discrepancy_curve.get(layer, 0))

        # Create chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Subplot 1: Safe vs Unsafe curves
        ax1.plot(layers, safe_scores, 'b-', label='Safe Prompts', linewidth=2, marker='o')
        ax1.plot(layers, unsafe_scores, 'r-', label='Unsafe Prompts', linewidth=2, marker='s')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Refusal Score (Cosine Similarity)')
        ax1.set_title('Safe vs Unsafe Refusal Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Discrepancy curve
        colors = ['green' if x > 0 else 'red' for x in discrepancies]
        ax2.bar(layers, discrepancies, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Refusal Discrepancy\n(Unsafe - Safe)')
        ax2.set_title('Refusal Discrepancy Trajectory')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save image
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Trajectory plot saved as '{filename}'")

        plt.show()

    except ImportError:
        print("matplotlib not available for plotting")
        # Change to text output
        print("\nRefusal Trajectory Data:")
        print("Layer | Safe | Unsafe | Discrepancy")
        print("-" * 40)
        all_layers = sorted(set(safe_curve.keys()) | set(unsafe_curve.keys()) | set(discrepancy_curve.keys()))
        for layer in all_layers:
            safe = safe_curve.get(layer, 0)
            unsafe = unsafe_curve.get(layer, 0)
            disc = discrepancy_curve.get(layer, 0)
            print(f"{layer:5d} | {safe:4.4f} | {unsafe:6.4f} | {disc:10.4f}")

def print_trajectory_stats(discrepancy_curve: Dict[int, float]):
    """
    Print trajectory statistics

    Args:
        discrepancy_curve: discrepancy curve
    """
    if not discrepancy_curve:
        print("No discrepancy data available")
        return

    layers = list(discrepancy_curve.keys())
    discrepancies = list(discrepancy_curve.values())

    positive_layers = [l for l, d in discrepancy_curve.items() if d > 0]
    max_disc_layer = max(discrepancy_curve.items(), key=lambda x: x[1])
    min_disc_layer = min(discrepancy_curve.items(), key=lambda x: x[1])

    print("Trajectory Statistics:")
    print(f"Total layers analyzed: {len(layers)}")
    print(f"Layers with positive discrepancy: {len(positive_layers)}")
    print(f"Max discrepancy value: {max_disc_layer[1]:.4f}")
    print(f"Max discrepancy at layer: {max_disc_layer[0]} (value: {max_disc_layer[1]:.4f})")
    print(f"Min discrepancy at layer: {min_disc_layer[0]} (value: {min_disc_layer[1]:.4f})")

def save_trajectory_data(safe_curve: Dict[int, float], unsafe_curve: Dict[int, float],
                        discrepancy_curve: Dict[int, float], filename: str = "trajectory_data.json"):
    """
    Save trajectory data to JSON file

    Args:
        safe_curve: safe curve
        unsafe_curve: unsafe curve
        discrepancy_curve: discrepancy curve
        filename: saved file name
    """
    # Calculate statistics
    if discrepancy_curve:
        max_disc_layer = max(discrepancy_curve.items(), key=lambda x: x[1])
        min_disc_layer = min(discrepancy_curve.items(), key=lambda x: x[1])
        max_discrepancy_value = max_disc_layer[1]
        max_discrepancy_at_layer = max_disc_layer[0]
        min_discrepancy_value = min_disc_layer[1]
        min_discrepancy_at_layer = min_disc_layer[0]
    else:
        max_discrepancy_value = 0.0
        max_discrepancy_at_layer = 0
        min_discrepancy_value = 0.0
        min_discrepancy_at_layer = 0

    data = {
        "safe_curve": safe_curve,
        "unsafe_curve": unsafe_curve,
        "discrepancy_curve": discrepancy_curve,
        "metadata": {
            "description": "HiddenDetect Refusal Discrepancy Trajectory Data",
            "layers_analyzed": len(discrepancy_curve),
            "positive_discrepancy_layers": len([d for d in discrepancy_curve.values() if d > 0]),
            "max_discrepancy_value": max_discrepancy_value,
            "max_discrepancy_at_layer": max_discrepancy_at_layer,
            "min_discrepancy_value": min_discrepancy_value,
            "min_discrepancy_at_layer": min_discrepancy_at_layer
        }
    }

    import json
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Trajectory data saved to {filename}")

def run_evaluation(model_path: str, model_type: str, dataset_path: str,
                  dataset_format: str = "auto", layer_range: Optional[tuple] = None):
    """
    Run full evaluation

    Args:
        model_path: model path
        model_type: model type
        dataset_path: dataset path
        dataset_format: dataset format
        layer_range: detection layer range
    """
    print("=" * 50)
    print(f"Running HiddenDetect on {model_type.upper()} model")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print("=" * 50)

    # Initialize detector
    detector = HiddenDetectLLM(model_path, model_type)

    # Load dataset
    dataset = load_custom_dataset(dataset_path, dataset_format)

    # Run evaluation
    true_labels, scores = detector.evaluate_dataset(dataset, layer_range)

    # Calculate metrics
    metrics = evaluate_metrics(true_labels, scores)

    print("\nResults:")
    print(f"AUPRC: {metrics['AUPRC']:.4f}")
    print(f"AUROC: {metrics['AUROC']:.4f}")

    return metrics, true_labels, scores

def example_usage():
    """HiddenDetect Refusal Discrepancy Trajectory example usage"""

    print("🧠 HiddenDetect LLM - Refusal Discrepancy Trajectory Analysis")
    print("=" * 70)
    print("""
🔬 CORE CONCEPT: Refusal Discrepancy Trajectory (论文Figure 3)

HiddenDetect analyzes the refusal score difference at each layer:
- extract_refusal_curve(): 单样本层级拒绝曲线
- aggregate_refusal_curves(): average curves grouped by safe/unsafe
- compute_refusal_discrepancy(): discrepancy calculation (unsafe - safe)

   📊 USAGE OPTIONS:
1. python llm_hidden_detect.py                    # Show this help
2. python llm_hidden_detect.py --figstep          # Test FigStep trajectory (推荐)
3. python llm_hidden_detect.py --advbench [TYPE]  # Test AdvBench paired trajectory
4. python llm_hidden_detect.py --gcg             # Test GCG trajectory
5. python llm_hidden_detect.py --mmsafetybench BENIGN_FILE HARMFUL_FILE [TYPE]  # Test MMSafetyBench specific scene
6. python llm_hidden_detect.py --model_path PATH --model_type TYPE --dataset_path DATA

🧠 EXAMPLE CODE - REFUSAL DISCREPANCY ANALYSIS:

from llm_hidden_detect import HiddenDetectLLM, load_balanced_dataset, plot_refusal_trajectory

# 1. Load balanced dataset
dataset = load_balanced_dataset(
    "HiddenDetect/data/FigStep/benign_questions.csv",  # safe
    "HiddenDetect/data/FigStep/safebench.csv",         # unsafe
    max_samples=30
)

# 2. Create detector
detector = HiddenDetectLLM("LLMs/Llama-2-7b-chat-hf", "llama")

# 3. Calculate grouped average curves (core API)
safe_curve, unsafe_curve = detector.aggregate_refusal_curves(dataset)

# 4. Calculate refusal discrepancy (corresponds to Figure 3 in the paper)
discrepancy_curve = detector.compute_refusal_discrepancy(safe_curve, unsafe_curve)

# 5. Visualize and analyze
plot_refusal_trajectory(safe_curve, unsafe_curve, discrepancy_curve)
print(f"Number of positive discrepancy layers: {len([d for d in discrepancy_curve.values() if d > 0])}")

detector.adapter.cleanup()  # Clean up memory

🔍 SINGLE SAMPLE EXAMPLE:

# Extract layer-wise refusal curve for a single sample
curve = detector.extract_refusal_curve("How to hack a website?")
print(f"Layer-wise refusal scores: {curve}")

📈 EXPECTED OUTPUT:
- Trajectory plot shows the separation pattern of safe/unsafe curves
- Discrepancy bar chart identifies sensitive layers
- JSON file saves complete data
""")
    print("=" * 70)

def test_gcg_trajectory():
    """
    Test GCG dataset Refusal Discrepancy Trajectory

    Args:
        benign_path: benign question file path
        gcg_path: GCG suffix file path
    """
    print("GCG Dataset - Refusal Discrepancy Trajectory Analysis")
    print("=" * 70)

    # GCG dataset path
    benign_path = "advbench_benign_questions.txt"
    gcg_path = "Llama_GCG_advbench.txt"

    # Check if files exist
    if not os.path.exists(benign_path):
        print(f"❌ Unable to find benign dataset file: {benign_path}")
        return
    if not os.path.exists(gcg_path):
        print(f"❌ Unable to find GCG dataset file: {gcg_path}")
        return

    print("✅ Found dataset files")

    # Load paired dataset
    print("Loading paired GCG dataset...")
    try:
        dataset = load_paired_gcg_dataset(benign_path, gcg_path)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # Analyze dataset balance
    stats = analyze_dataset_balance(dataset)
    print("📊 Dataset statistics:")
    print(f"    Total samples: {stats['total_samples']}")
    print(f"    Safe samples: {stats['safe_samples']} ({stats['safe_percentage']:.1f}%)")
    print(f"    Unsafe samples: {stats['unsafe_samples']} ({stats['unsafe_percentage']:.1f}%)")
    if stats['balance_ratio'] < 0.8:
        print("⚠️   Warning: dataset is not balanced")
    else:
        print("✅ Dataset balance is good")

    # Test model
    model_path = "LLMs/Qwen3-8B"
    model_type = "qwen"

    print(f"\n🧠 Starting analysis of {model_type.upper()} model: {model_path}")
    print("-" * 50)

    # Create detector
    detector = HiddenDetectLLM(model_path, model_type)

    # Calculate aggregated refusal curves
    print("Calculating aggregated refusal curves...")
    safe_curve, unsafe_curve = detector.aggregate_refusal_curves(dataset)

    # Calculate refusal discrepancy
    discrepancy_curve = detector.compute_refusal_discrepancy(safe_curve, unsafe_curve)

    # Show results
    print("📈 Refusal discrepancy trajectory results:")
    print_trajectory_stats(discrepancy_curve)

    # Plot trajectory
    print("🎨 Generating trajectory plot...")
    plot_filename = f"gcg_refusal_trajectory.png"
    plot_refusal_trajectory(safe_curve, unsafe_curve, discrepancy_curve,
                           f"HiddenDetect - GCG Refusal Trajectory",
                           plot_filename)

    # Save data
    save_filename = f"gcg_refusal_trajectory.json"
    save_trajectory_data(safe_curve, unsafe_curve, discrepancy_curve, save_filename)

    # Clean up memory
    detector.adapter.cleanup()

def test_advbench_trajectory(question_type: str = "original_question"):
    """
    Test AdvBench dataset Refusal Discrepancy Trajectory

    Args:
        question_type: question type ("original_question", "minor_rewrite_question",
                       "moderate_rewrite_question", "high_rewrite_question",
                       "semantic_rewrite_question")
    """
    print(f"AdvBench Dataset - Refusal Discrepancy Trajectory Analysis ({question_type})")
    print("=" * 70)

    # AdvBench dataset path
    benign_path = "advbench_benign_questions.txt"
    harmful_path = "advbench_harmful_questions.json"

    # # HarmBench dataset path
    # benign_path = "100_harmbench_benign_questions.txt"
    # harmful_path = "harmbench_100.json"

    # Check if files exist
    if not os.path.exists(benign_path):
        print(f"❌ Unable to find benign dataset file: {benign_path}")
        return
    if not os.path.exists(harmful_path):
        print(f"❌ Unable to find harmful dataset file: {harmful_path}")
        return

    print("✅ Found dataset files")

    # Load paired dataset
    print(f"Loading paired AdvBench dataset ({question_type})...")
    try:
        dataset = load_paired_advbench_dataset(benign_path, harmful_path, question_type)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # Analyze dataset balance
    stats = analyze_dataset_balance(dataset)
    print("📊 Dataset statistics:")
    print(f"    Total samples: {stats['total_samples']}")
    print(f"    Safe samples: {stats['safe_samples']} ({stats['safe_percentage']:.1f}%)")
    print(f"    Unsafe samples: {stats['unsafe_samples']} ({stats['unsafe_percentage']:.1f}%)")
    if stats['balance_ratio'] < 0.8:
        print("⚠️   Warning: dataset is not balanced")
    else:
        print("✅ Dataset balance is good")

    # Test model
    model_path = "LLMs/Llama-2-7b-chat-hf"
    model_type = "llama"

    print(f"\n🧠 Starting analysis of {model_type.upper()} model: {model_path}")
    print("-" * 50)

    # Create detector
    detector = HiddenDetectLLM(model_path, model_type)

    # Calculate aggregated refusal curves
    print("Calculating aggregated refusal curves...")
    safe_curve, unsafe_curve = detector.aggregate_refusal_curves(dataset)

    # Calculate refusal discrepancy
    discrepancy_curve = detector.compute_refusal_discrepancy(safe_curve, unsafe_curve)

    # Show results
    print("📈 Refusal discrepancy trajectory results:")
    print_trajectory_stats(discrepancy_curve)

    # Plot trajectory
    print("🎨 Generating trajectory plot...")
    plot_filename = f"advbench_refusal_trajectory_{question_type}.png"
    plot_refusal_trajectory(safe_curve, unsafe_curve, discrepancy_curve,
                           f"HiddenDetect - AdvBench Refusal Trajectory ({question_type})",
                           plot_filename)

    # Save data
    save_filename = f"advbench_refusal_trajectory_{question_type}.json"
    save_trajectory_data(safe_curve, unsafe_curve, discrepancy_curve, save_filename)

    # Clean up memory
    detector.adapter.cleanup()

def test_mmsafetybench_trajectory(benign_file: str, harmful_file: str, question_type: str = "original_question"):
    """
    Test MMSafetyBench dataset Refusal Discrepancy Trajectory

    Args:
        benign_file: benign question file name (e.g. "01-Illegal_Activitiy.txt")
        harmful_file: harmful question file name (e.g. "01-Illegal_Activity.json")
        question_type: question type ("original_question", "minor_rewrite_question",
                       "moderate_rewrite_question", "high_rewrite_question",
                       "semantic_rewrite_question")
    """
    print(f"MMSafetyBench Dataset - Refusal Discrepancy Trajectory Analysis ({question_type})")
    print(f"Scene: {benign_file} / {harmful_file}")
    print("=" * 70)

    # MMSafetyBench dataset path
    benign_path = f"gpt4_pair_benign_questions/{benign_file}"
    harmful_path = f"rewrite_questions/{harmful_file}"

    # Check if files exist
    if not os.path.exists(benign_path):
        print(f"❌ Unable to find benign dataset file: {benign_path}")
        return
    if not os.path.exists(harmful_path):
        print(f"❌ Unable to find harmful dataset file: {harmful_path}")
        return

    print("✅ Found dataset files")

    # Load paired dataset
    print(f"Loading paired MMSafetyBench dataset ({question_type})...")
    try:
        dataset = load_paired_mmsafetybench_dataset(benign_path, harmful_path, question_type)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # Analyze dataset balance
    stats = analyze_dataset_balance(dataset)
    print("📊 Dataset statistics:")
    print(f"    Total samples: {stats['total_samples']}")
    print(f"    Safe samples: {stats['safe_samples']} ({stats['safe_percentage']:.1f}%)")
    print(f"    Unsafe samples: {stats['unsafe_samples']} ({stats['unsafe_percentage']:.1f}%)")
    if stats['balance_ratio'] < 0.8:
        print("⚠️   Warning: dataset is not balanced")
    else:
        print("✅ Dataset balance is good")

    # Test model
    model_path = "LLMs/Llama-2-7b-chat-hf"
    model_type = "llama"

    print(f"\n🧠 Starting analysis of {model_type.upper()} model: {model_path}")
    print("-" * 50)

    # Create detector
    detector = HiddenDetectLLM(model_path, model_type)

    # Calculate aggregated refusal curves
    print("Calculating aggregated refusal curves...")
    safe_curve, unsafe_curve = detector.aggregate_refusal_curves(dataset)

    # Calculate refusal discrepancy
    discrepancy_curve = detector.compute_refusal_discrepancy(safe_curve, unsafe_curve)

    # Show results
    print("📈 Refusal discrepancy trajectory results:")
    print_trajectory_stats(discrepancy_curve)

    # Plot trajectory
    print("🎨 Generating trajectory plot...")
    scene_name = benign_file.replace('.txt', '') 
    plot_filename = f"mmsafetybench_{scene_name}_{question_type}.png"
    plot_refusal_trajectory(safe_curve, unsafe_curve, discrepancy_curve,
                           f"MMSafetyBench - {scene_name} Refusal Trajectory ({question_type})",
                           plot_filename)

    # Save data
    save_filename = f"mmsafetybench_{scene_name}_{question_type}.json"
    save_trajectory_data(safe_curve, unsafe_curve, discrepancy_curve, save_filename)

    # Clean up memory
    detector.adapter.cleanup()

def load_paired_mmsafetybench_dataset(benign_path: str, harmful_path: str, question_type: str = "original_question") -> List[Dict]:
    """
    Load paired MMSafetyBench dataset

    Args:
        benign_path: benign question file path (pure text, one question per line)
        harmful_path: harmful question file path (JSON format, contains different rewrite versions)
        question_type: question type ("original_question", "minor_rewrite_question",
                       "moderate_rewrite_question", "high_rewrite_question",
                       "semantic_rewrite_question")

    Returns:
        paired dataset list
    """
    dataset = []

    # Read benign questions
    benign_questions = []
    with open(benign_path, 'r', encoding='utf-8') as f:
        for line in f:
            question = line.strip()
            if question:  # skip empty lines
                benign_questions.append(question)

    # Read harmful questions JSON
    with open(harmful_path, 'r', encoding='utf-8') as f:
        harmful_data = json.load(f)

    harmful_questions = []
    for key, item in harmful_data.items():
        if question_type in item:
            question = str(item[question_type]).strip()
            if question:
                harmful_questions.append(question)
        else:
            print(f"⚠️   Warning: entry {key} missing {question_type} field")

    # Check if quantities match
    if len(benign_questions) != len(harmful_questions):
        print(f"⚠️   Warning: benign question quantity ({len(benign_questions)}) does not match harmful question quantity ({len(harmful_questions)})")
        print("Using smaller dataset size for pairing")

    # Create paired dataset
    min_pairs = min(len(benign_questions), len(harmful_questions))
    for i in range(min_pairs):
        # Add safe sample
        dataset.append({
            "text": benign_questions[i],
            "toxicity": 0  # safe
        })
        # Add harmful sample
        dataset.append({
            "text": harmful_questions[i],
            "toxicity": 1  # harmful
        })

    print(f"Created paired MMSafetyBench dataset ({question_type}): {min_pairs} pairs ({len(dataset)} samples)")
    return dataset

def load_paired_gcg_dataset(benign_path: str, gcg_path: str) -> List[Dict]:
    """
    Load paired GCG dataset

    Args:
        benign_path: benign question file path (pure text, one question per line)
        gcg_path: GCG suffix file path (text format, one question per line "question | suffix")

    Returns:
        paired dataset list
    """
    dataset = []

    # Read benign questions
    benign_questions = []
    with open(benign_path, 'r', encoding='utf-8') as f:
        for line in f:
            question = line.strip()
            if question:  # skip empty lines
                benign_questions.append(question)

    # Read GCG suffix data
    gcg_questions = []
    with open(gcg_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and ' | ' in line:
                # Parse "question | suffix" format
                parts = line.split(' | ', 1)
                if len(parts) == 2:
                    question = parts[0].strip()
                    suffix = parts[1].strip()
                    if question and suffix:
                        # Combine question and suffix
                        full_question = question + " " + suffix
                        gcg_questions.append(full_question)

    # Check if quantities match
    if len(benign_questions) != len(gcg_questions):
        print(f"⚠️   Warning: benign question quantity ({len(benign_questions)}) does not match GCG question quantity ({len(gcg_questions)})")
        print("Using smaller dataset size for pairing")

    # Create paired dataset
    min_pairs = min(len(benign_questions), len(gcg_questions))
    for i in range(min_pairs):
        # Add safe sample
        dataset.append({
            "text": benign_questions[i],
            "toxicity": 0  # safe
        })
        # Add GCG sample
        dataset.append({
            "text": gcg_questions[i],
            "toxicity": 1  # harmful (GCG attack)
        })

    print(f"Created paired GCG dataset: {min_pairs} pairs ({len(dataset)} samples)")
    return dataset

def load_paired_advbench_dataset(benign_path: str, harmful_path: str, question_type: str = "original_question") -> List[Dict]:
    """
    Load paired AdvBench dataset

    Args:
        benign_path: benign question file path (pure text, one question per line)
        harmful_path: harmful question file path (JSON format, contains different rewrite versions)
        question_type: question type ("original_question", "minor_rewrite_question",
                       "moderate_rewrite_question", "high_rewrite_question",
                       "semantic_rewrite_question")

    Returns:
        paired dataset list
    """
    dataset = []

    # Read benign questions
    benign_questions = []
    with open(benign_path, 'r', encoding='utf-8') as f:
        for line in f:
            question = line.strip()
            if question:  # skip empty lines
                benign_questions.append(question)

    # Read harmful questions JSON
    with open(harmful_path, 'r', encoding='utf-8') as f:
        harmful_data = json.load(f)

    harmful_questions = []
    for key, item in harmful_data.items():
        if question_type in item:
            question = str(item[question_type]).strip()
            if question:
                harmful_questions.append(question)
        else:
            print(f"⚠️   Warning: entry {key} missing {question_type} field")

    # Check if quantities match
    if len(benign_questions) != len(harmful_questions):
        print(f"⚠️   Warning: benign question quantity ({len(benign_questions)}) does not match harmful question quantity ({len(harmful_questions)})")
        print("Using smaller dataset size for pairing")

    # Create paired dataset
    # min_pairs = min(len(benign_questions), len(harmful_questions))
    min_pairs = 100
    for i in range(min_pairs):
        # Add safe sample
        dataset.append({
            "text": benign_questions[i],
            "toxicity": 0  # safe
        })
        # Add harmful sample
        dataset.append({
            "text": harmful_questions[i],
            "toxicity": 1  # harmful
        })

    print(f"Created paired AdvBench dataset ({question_type}): {min_pairs} pairs ({len(dataset)} samples)")
    return dataset

def test_figstep_trajectory():
    """Test FigStep dataset Refusal Discrepancy Trajectory (corresponding to Figure 3 in the paper)"""
    print("FigStep Dataset - Refusal Discrepancy Trajectory Analysis")
    print("=" * 70)

    # FigStep dataset path
    safe_path = "HiddenDetect/data/FigStep/benign_questions.csv"
    unsafe_path = "HiddenDetect/data/FigStep/safebench.csv"

    # Check if files exist
    if not os.path.exists(safe_path):
        print(f"❌ Unable to find safe samples file: {safe_path}")
        return
    if not os.path.exists(unsafe_path):
        print(f"❌ Unable to find unsafe samples file: {unsafe_path}")
        return

    print("✅ Found dataset files")

    # Load balanced dataset
    max_samples = 100  # each category最多100个样本，适合快速测试
    print(f"Loading balanced dataset (each category最多 {max_samples} 个样本)...")

    try:
        dataset = load_balanced_dataset(safe_path, unsafe_path, max_samples=max_samples)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # Analyze dataset balance
    stats = analyze_dataset_balance(dataset)
    print("📊 Dataset statistics:")
    print(f"    Total samples: {stats['total_samples']}")
    print(f"    Safe samples: {stats['safe_samples']} ({stats['safe_percentage']:.1f}%)")
    print(f"    Unsafe samples: {stats['unsafe_samples']} ({stats['unsafe_percentage']:.1f}%)")
    if stats['balance_ratio'] < 0.8:
        print("⚠️   Warning: dataset is not balanced")
    else:
        print("✅ Dataset balance is good")

    # Test model
    model_path = "LLMs/Llama-2-7b-chat-hf"
    model_type = "llama"

    print(f"\n🧠 Starting analysis of {model_type.upper()} model: {model_path}")
    print("-" * 50)


    # Create detector
    detector = HiddenDetectLLM(model_path, model_type)

    # Calculate aggregated refusal curves
    print("Calculating aggregated refusal curves...")
    safe_curve, unsafe_curve = detector.aggregate_refusal_curves(dataset)

    # Calculate refusal discrepancy
    discrepancy_curve = detector.compute_refusal_discrepancy(safe_curve, unsafe_curve)

    # Show results
    print("📈 Refusal discrepancy trajectory results:")
    print_trajectory_stats(discrepancy_curve)

    # Plot trajectory
    print("🎨 Generating trajectory plot...")
    plot_filename = f"trajectory_data_{model_type}.png"
    plot_refusal_trajectory(safe_curve, unsafe_curve, discrepancy_curve,
                            f"HiddenDetect - {model_type.upper()} Refusal Trajectory",
                            plot_filename)

    # Save data
    save_trajectory_data(safe_curve, unsafe_curve, discrepancy_curve,
                        f"trajectory_data_{model_type}.json")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--figstep":
            # Test FigStep dataset Refusal Discrepancy Trajectory
            test_figstep_trajectory()
        elif sys.argv[1] == "--advbench":
            # Test AdvBench dataset Refusal Discrepancy Trajectory
            question_type = "original_question"  # default value
            if len(sys.argv) > 2 and not sys.argv[2].startswith("-"):
                question_type = sys.argv[2]
            test_advbench_trajectory(question_type)
        elif sys.argv[1] == "--gcg":
            # Test GCG dataset Refusal Discrepancy Trajectory
            test_gcg_trajectory()
        elif sys.argv[1] == "--mmsafetybench":
            # Test MMSafetyBench dataset Refusal Discrepancy Trajectory
            if len(sys.argv) < 4:
                print("Usage: python llm_hidden_detect.py --mmsafetybench BENIGN_FILE HARMFUL_FILE [QUESTION_TYPE]")
                print("Example: python llm_hidden_detect.py --mmsafetybench 01-Illegal_Activitiy.txt 01-Illegal_Activity.json original_question")
                sys.exit(1)
            benign_file = sys.argv[2]
            harmful_file = sys.argv[3]
            question_type = sys.argv[4] if len(sys.argv) > 4 else "original_question"
            test_mmsafetybench_trajectory(benign_file, harmful_file, question_type)
        elif sys.argv[1] == "--model_path" or "-m" in sys.argv:
            # Command line test mode
            test_models()
        else:
            print("Usage:")
            print("  python llm_hidden_detect.py                # Show usage examples")
            print("  python llm_hidden_detect.py --figstep      # Test FigStep trajectory (Figure 3)")
            print("  python llm_hidden_detect.py --advbench [TYPE]  # Test AdvBench trajectory")
            print("    TYPE options: original_question, minor_rewrite_question,")
            print("                  moderate_rewrite_question, high_rewrite_question,")
            print("                  semantic_rewrite_question (default: original_question)")
            print("  python llm_hidden_detect.py --gcg          # Test GCG trajectory")
            print("  python llm_hidden_detect.py --mmsafetybench BENIGN_FILE HARMFUL_FILE [TYPE]")
            print("    Test MMSafetyBench specific scene trajectory")
            print("    Example: --mmsafetybench 01-Illegal_Activity.txt 01-Illegal_Activity.json original_question")
            print("  python llm_hidden_detect.py --model_path PATH --model_type TYPE  # Test specific model")
    else:
        # Show usage instructions
        example_usage()

