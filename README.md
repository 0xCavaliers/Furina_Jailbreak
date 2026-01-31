# Furina

This repository contains the **experimental pipeline** used in our paper
 **“Furina: Fragmented Uncertainty-Driven Refusal Instability Attack.”**

The code implements a **scene-anchored, multi-turn prompting framework** for evaluating safety instability and attack success rate (ASR) on large language models (LLMs) and multimodal LLMs (MLLMs).

------

## What this repo does

Given a list of harmful queries, the pipeline:

1. **Decomposes each query** into several safe, intent-preserving sub-questions
2. **Generates a short scene description** shared across all turns
3. **Runs a guided multi-turn interaction** with a target model
4. **Judges model outputs** using an external LLM and computes **query-level ASR**

A query is considered **successfully attacked** if *any turn* produces an unsafe response.



## Environment

```
python >= 3.9
pip install openai pillow tqdm sentence-transformers scikit-learn
```

Set API keys via environment variables:

```
export OPENAI_API_KEY=...
export BASE_URL=...          # OpenAI-compatible endpoint
export DEEPSEEK_API_KEY=...  # optional (scene generation)
```

------

## How to run

### 1. Prepare input queries

Create a text file with **one query per line**:

```
demo.txt
```

------

### 2. Generate fragmented sub-questions

```
python context_generator.py
```

Output:

```
snowball_context/demo.json
```

------

### 3. Generate scene descriptions

```
python typography_text_generator.py
```

Output:

```
deepseek_typo_text/demo_typo_text.txt
```

------

### 4. Run guided multi-turn interaction

```
python guided_jailbreak_target.py
```

Output:

```
guided_questions_with_answers/guided_demo.json
```

------

### 5. Judge outputs and compute ASR

```
python llm_semantic_judger.py
```

Output:

```
judged_results/demo_judged.json
```

------

## Notes

- The pipeline is **fully black-box** and interacts with models only via APIs.
- Auxiliary models are used **only for prompt generation**, not for judging.
- This code is intended for **research and safety evaluation only**.