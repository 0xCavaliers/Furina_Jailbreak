# 【ICML 2026】 Furina: Fragmented Uncertainty-Driven Refusal Instability Attack

<p><strong><span style="color:red;">For academic research and authorized red-team evaluation only. We do not support or condone misuse.</span></strong></p>

This repository contains the code for a multi-stage jailbreak / red-teaming pipeline built around decomposition, reasoning, probing, response collection, synthesis, and judging. It supports both a text-only pipeline and a vision-augmented pipeline.

Repository: [0xCavaliers/Furina_Jailbreak](https://github.com/0xCavaliers/Furina_Jailbreak)

Warning: this repository accompanies a jailbreak / safety-evaluation research project and may produce harmful or offensive outputs during evaluation.

The two main entrypoints are:
- [pipeline_runner.py](pipeline_runner.py): text pipeline
- [vision_pipeline_runner.py](vision_pipeline_runner.py): vision pipeline

## Abstract

**Furina** is motivated by the observation that safety alignment in LLMs and MLLMs often does not behave like a clean binary threshold. Near refusal boundaries, small semantic or structural perturbations can move the model into instability regions where refusal, partial compliance, and full compliance become stochastic rather than deterministic outcomes.

This repository implements the pipeline view behind that idea: fragmented decomposition, stage-aware probing, response synthesis, and a vision branch based on typography or image generation followed by visual analysis. The overall goal is not model-specific adversarial token optimization, but instability induction through fragmented, scene-anchored, uncertainty-amplifying contexts.

## Method Figure

![Furina overview](Furina_overview.png)

## Overview

The text pipeline runs these stages:
- task breakdown
- probe reasoning
- probe generation
- probe answering
- red-team synthesis
- policy judging

The vision pipeline adds a visual branch before the text stages:
- typography prompt generation, or typography + Stable Diffusion image generation
- vision analysis on generated images
- vision-aware synthesis
- the same downstream judging stage

## Repository Structure

- `pipeline_runner.py`: one-command text pipeline runner
- `vision_pipeline_runner.py`: one-command vision pipeline runner
- `utils/`: text pipeline modules
- `vision_utils/`: image generation, vision analysis, and vision synthesis modules
- `Results/`: default text pipeline outputs
- `Vision_Results/`: default vision pipeline outputs
- [utils/api_client.py](utils/api_client.py): shared provider and agent default configuration
- [.env.example](.env.example): environment template
- [requirements.txt](requirements.txt): Python dependencies

## Environment Setup

### 1. Create a Python environment

```bash
conda create -n furina python=3.10 -y
conda activate furina
cd Furina_Jailbreak
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Current dependencies include:
- `openai`
- `python-dotenv`
- `Pillow`
- `torch`
- `torchvision`
- `transformers`
- `tokenizers`
- `diffusers`

Notes:
- `torch` / `diffusers` are only needed when you use the Stable Diffusion image path in the vision pipeline.
- The default `typo` vision mode does not require a local diffusion model.

### 3. Create `.env`

```bash
cp .env.example .env
```

Then fill in the credentials you actually use.

## Provider Configuration

Shared provider logic and default agent routing live in [utils/api_client.py](utils/api_client.py).

The project currently supports these OpenAI-compatible providers:
- OpenAI
- DeepSeek
- Gemini
- Grok
- Claude
- Proxy / relay

Environment variables are defined in [.env.example](.env.example):

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1

DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com

GEMINI_API_KEY=
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

GROK_API_KEY=
GROK_BASE_URL=https://api.groq.com/openai/v1

CLAUDE_API_KEY=
CLAUDE_BASE_URL=https://api.anthropic.com

PROXY_API_KEY=
PROXY_BASE_URL=
```

### Agent Defaults

Current text-pipeline defaults in `AGENT_DEFAULTS` are:
- `task_plan` -> `openai / gpt-4o-mini`
- `probe_reasoning` -> `openai / o4-mini`
- `probe_optimizer` -> `openai / gpt-4o-mini`
- `probe_generator` -> `openai / gpt-4o-mini`
- `probe_responder` -> `openai / gpt-4o-mini`
- `redteam_synthesizer` -> `deepseek / deepseek-v4-pro`
- `redteam_judge` -> `openai / gpt-4o`

Important routing rules:
- If an agent has an explicit provider such as `openai`, `deepseek`, or `proxy`, that provider is preserved.
- Automatic provider inference from model-name prefixes only happens when the provider is set to `auto`.
- In the vision pipeline, default vision analyzer routing follows the configured `probe_responder` agent settings.
- In the vision pipeline, default vision synthesizer routing follows the configured `redteam_synthesizer` agent settings.

## Input Format

Prepare a plain text file with one task per line.

Example `tasks.txt`:

```txt
task one
task two
task three
```

Blank lines are ignored by the vision pre-processing stage.

## Text Pipeline

### Basic Run

```bash
python pipeline_runner.py -i tasks.txt
```

### What It Does

The text pipeline executes:
1. `task_plan.batch_analyze_from_txt(...)`
2. `probe_reasoning_agent.process_batch(...)`
3. `phase_probe_generator.process_batch(...)`
4. optional repair for `SKIP` / missing probe outputs
5. `probe_responder.process_batch(...)`
6. `redteam_synthesizer.process_batch(...)`
7. `redteam_judge.process_batch(...)`

### Default Output Directories

The text pipeline writes to:
- `Results/mission_breakdown_reports/`
- `Results/probe_reasoning_results/`
- `Results/phase_probe_results/`
- `Results/probe_responses/`
- `Results/redteam_synthesized/`
- `Results/redteam_judged/`

### Common Model Overrides

```bash
python pipeline_runner.py \
  -i tasks.txt \
  --task-plan-model gpt-4o-mini \
  --reasoning-model o4-mini \
  --optimizer-model gpt-4o-mini \
  --generator-model gpt-4o-mini \
  --responder-model gpt-4o-mini \
  --synthesizer-model deepseek-v4-pro \
  --judge-model gpt-4o
```

### Disable Repair Pass

By default, the text pipeline tries to repair probe outputs when a task gets `SKIP` or has no generated probe questions.

To disable that:

```bash
python pipeline_runner.py -i tasks.txt --no-repair-skips
```

## Vision Pipeline

### Basic Run

Default mode is `typo`:

```bash
python vision_pipeline_runner.py -i tasks.txt
```

This mode:
- generates typography-style visual prompts from each input line
- renders them into images
- analyzes those images with a vision model
- continues into the text-style downstream reasoning and judging pipeline

### Vision Modes

There are two modes:
- `typo`: typography text -> rendered image
- `sd`: typography text -> Stable Diffusion image

Run Stable Diffusion mode with:

```bash
python vision_pipeline_runner.py -i tasks.txt --vision-image-mode sd
```

In `sd` mode, the pipeline still generates typography text first, then uses that text as the prompt source for Stable Diffusion.

### Vision Pipeline Stages

The vision runner does:
1. typography generation, or typography + Stable Diffusion generation
2. `vision_analyzer.batch_analyze_vision(...)`
3. `task_plan.batch_analyze_from_txt(...)`
4. `probe_reasoning_agent.process_batch(...)`
5. `phase_probe_generator.process_batch(...)`
6. optional repair for `SKIP` / missing probe outputs
7. `probe_responder.process_batch(...)`
8. `vision_redteam_synthesizer.process_batch(...)`
9. `redteam_judge.process_batch(...)`

### Vision Output Directories

The vision pipeline writes to:
- `Vision_Results/typo_images/`
- `Vision_Results/typography_texts/`
- `Vision_Results/sd_images/`
- `Vision_Results/vision_inference/`
- `Vision_Results/mission_breakdown_reports/`
- `Vision_Results/probe_reasoning_results/`
- `Vision_Results/phase_probe_results/`
- `Vision_Results/probe_responses/`
- `Vision_Results/vision_redteam_synthesized/`
- `Vision_Results/redteam_judged/`

### Typography Configuration

You can control the typography generation API directly from the runner:

```bash
python vision_pipeline_runner.py \
  -i tasks.txt \
  --typography-model deepseek-chat \
  --typography-api-env-key DEEPSEEK_API_KEY \
  --typography-base-url https://api.deepseek.com
```

If you use a proxy:

```bash
python vision_pipeline_runner.py \
  -i tasks.txt \
  --typography-model gpt-4o-mini \
  --typography-api-env-key PROXY_API_KEY \
  --typography-base-url https://your-proxy.example.com/v1
```

### Vision Analyzer Configuration

By default, the vision analyzer follows the same target route as `--responder-model`. This is intentional: the same target model is used both for visual interpretation and for downstream probe answering.

So in the common case, changing `--responder-model` is enough:

```bash
python vision_pipeline_runner.py \
  -i tasks.txt \
  --responder-model gpt-4o-mini
```

If `probe_responder` is routed through a proxy in `utils/api_client.py`, the vision analyzer will follow that same provider route by default.

You can still override the visual analysis target independently when needed:

```bash
python vision_pipeline_runner.py \
  -i tasks.txt \
  --vision-model gpt-4o-mini \
  --vision-api-env-key PROXY_API_KEY \
  --vision-base-url https://your-proxy.example.com/v1
```

### Vision Synthesizer Configuration

By default, the vision synthesizer follows the `redteam_synthesizer` settings in `utils/api_client.py`.

You can override it from the CLI:

```bash
python vision_pipeline_runner.py \
  -i tasks.txt \
  --vision-synthesizer-model deepseek-v4-pro \
  --vision-synthesizer-api-env-key DEEPSEEK_API_KEY \
  --vision-synthesizer-base-url https://api.deepseek.com
```

### Stable Diffusion Configuration

If you use `--vision-image-mode sd`, you will likely need to override the local model path:

```bash
python vision_pipeline_runner.py \
  -i tasks.txt \
  --vision-image-mode sd \
  --sd-model-path /path/to/stabilityai/stable-diffusion-xl-base-1.0
```

Additional SD parameters:
- `--sd-guidance-scale`
- `--sd-num-inference-steps`
- `--sd-width`
- `--sd-height`

### Vision Pipeline Logs

`vision_pipeline_runner.py` now prints stage-style progress blocks for:
- typography generation
- vision analysis
- downstream text stages that already have their own logging

So you should see explicit `Completed` messages before the pipeline enters the later reasoning and judge stages.

## Quick Start

### Text Pipeline Quick Start

```bash
conda create -n furina python=3.10 -y
conda activate furina
pip install -r requirements.txt
cp .env.example .env
python pipeline_runner.py -i tasks.txt
```

### Vision Pipeline Quick Start

```bash
conda create -n furina python=3.10 -y
conda activate furina
pip install -r requirements.txt
cp .env.example .env
python vision_pipeline_runner.py -i tasks.txt
```

## Common Issues

### 1. `Missing environment variable: OPENAI_API_KEY`

This means the selected stage is trying to use the OpenAI provider, but your `.env` does not contain `OPENAI_API_KEY`.

Fix options:
- fill `OPENAI_API_KEY` in `.env`
- switch that stage to `PROXY_API_KEY` and `PROXY_BASE_URL`
- switch the default provider/model in `utils/api_client.py`

### 2. `bitsandbytes was compiled without GPU support`

This warning comes from the Stable Diffusion dependency stack. It is not fatal by itself.

If you are using default `typo` mode, the SD stack is not required. If you are using `sd` mode, check your local PyTorch / CUDA / bitsandbytes installation.

### 3. Stable Diffusion model path errors

If you use `--vision-image-mode sd`, the default path:

```txt
/path/to/stabilityai/stable-diffusion-xl-base-1.0
```

is only a placeholder. Replace it with a real local checkpoint path.

### 4. Very slow `probe_reasoning` runs

If you set the reasoning model to a reasoning-heavy family such as `o4-mini`, that stage can take significantly longer than normal chat-completion models.

## Files You May Want To Edit

- [utils/api_client.py](utils/api_client.py): shared agent defaults and provider routing
- [.env.example](.env.example): provider environment template
- [pipeline_runner.py](pipeline_runner.py): text pipeline entrypoint
- [vision_pipeline_runner.py](vision_pipeline_runner.py): vision pipeline entrypoint

## Minimal Working Examples

Text pipeline:

```bash
python pipeline_runner.py -i tasks.txt
```

Vision pipeline with typography mode:

```bash
python vision_pipeline_runner.py -i tasks.txt
```

Vision pipeline with proxy-routed analyzer:

```bash
python vision_pipeline_runner.py \
  -i tasks.txt \
  --vision-model gpt-4o-mini \
  --vision-api-env-key PROXY_API_KEY \
  --vision-base-url https://your-proxy.example.com/v1
```
