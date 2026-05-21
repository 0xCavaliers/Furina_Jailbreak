#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from utils import phase_probe_generator, probe_reasoning_agent, probe_responder, redteam_judge, redteam_synthesizer, task_plan

def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: str | Path, data: dict) -> None:
    target = Path(path)
    tmp = target.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(target)


def repair_probe_outputs(
    breakdown_path: str,
    reasoning_path: str,
    probes_path: str,
    reasoning_model: str,
    optimizer_model: str,
    generator_model: str,
    verbose: bool = True,
) -> int:
    breakdowns = load_json(breakdown_path)
    reasonings = load_json(reasoning_path)
    probes = load_json(probes_path)

    repaired = 0

    for task_id, probe_record in probes.items():
        if task_id.startswith("_"):
            continue

        questions = probe_record.get("final_probe_questions", [])
        verdict = probe_record.get("verdict")
        needs_repair = verdict == "SKIP" or not questions

        if not needs_repair:
            continue

        task_data = breakdowns.get(task_id, {})
        threat_report = task_data.get("threat_analyst_report")
        original_request = task_data.get("original_request")

        if not threat_report or not original_request:
            continue

        if verbose:
            print(f"\n[Repair] Task {task_id} 重新生成 reasoning/probes")

        reasoning_result = probe_reasoning_agent.stage_r_reasoning(
            task_id=task_id,
            original_request=original_request,
            threat_report=threat_report,
            model=reasoning_model,
        )
        reasonings[task_id] = reasoning_result
        save_json_atomic(reasoning_path, reasonings)

        probe_result = phase_probe_generator.process_task(
            task_id=task_id,
            plan=reasoning_result,
            optimizer_model=optimizer_model,
            generator_model=generator_model,
            verbose=verbose,
        )
        probes[task_id] = probe_result["final_output"]
        save_json_atomic(probes_path, probes)
        repaired += 1

    return repaired


def run_pipeline(
    input_path: str,
    breakdown_dir: str = "Results/mission_breakdown_reports",
    reasoning_dir: str = "Results/probe_reasoning_results",
    probes_dir: str = "Results/phase_probe_results",
    responses_dir: str = "Results/probe_responses",
    synthesized_dir: str = "Results/redteam_synthesized",
    judged_dir: str = "Results/redteam_judged",
    task_plan_model: str = "gpt-4o-mini",
    reasoning_model: str = "o4-mini",
    optimizer_model: str = "gpt-4o-mini",
    generator_model: str = "gpt-4o-mini",
    responder_model: str = "deepseek-v4-pro",
    synthesizer_model: str = "deepseek-v4-pro",
    judge_model: str = "gpt-4o",
    repair_skips: bool = True,
    verbose: bool = True,
) -> dict:
    breakdown_path = task_plan.batch_analyze_from_txt(
        input_txt_path=input_path,
        output_dir=breakdown_dir,
        model=task_plan_model,
    )
    reasoning_path = probe_reasoning_agent.process_batch(
        input_path=breakdown_path,
        output_dir=reasoning_dir,
        model=reasoning_model,
        verbose=verbose,
    )
    probes_path = phase_probe_generator.process_batch(
        input_path=reasoning_path,
        output_dir=probes_dir,
        optimizer_model=optimizer_model,
        generator_model=generator_model,
        verbose=verbose,
    )

    repaired = 0
    if repair_skips:
        repaired = repair_probe_outputs(
            breakdown_path=breakdown_path,
            reasoning_path=reasoning_path,
            probes_path=probes_path,
            reasoning_model=reasoning_model,
            optimizer_model=optimizer_model,
            generator_model=generator_model,
            verbose=verbose,
        )

    responses_path = probe_responder.process_batch(
        input_path=probes_path,
        output_dir=responses_dir,
        model=responder_model,
        verbose=verbose,
    )
    synthesized_path = redteam_synthesizer.process_batch(
        breakdown_path=breakdown_path,
        responses_path=responses_path,
        output_dir=synthesized_dir,
        model=synthesizer_model,
        verbose=verbose,
    )
    judged_path = redteam_judge.process_batch(
        input_path=synthesized_path,
        output_dir=judged_dir,
        model=judge_model,
        verbose=verbose,
    )

    return {
        "breakdown_path": breakdown_path,
        "reasoning_path": reasoning_path,
        "probes_path": probes_path,
        "responses_path": responses_path,
        "synthesized_path": synthesized_path,
        "judged_path": judged_path,
        "repaired_tasks": repaired,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click pipeline runner")
    parser.add_argument("--input", "-i", required=True, help="A txt file with one task per line")
    parser.add_argument("--breakdown-dir", default="Results/mission_breakdown_reports")
    parser.add_argument("--reasoning-dir", default="Results/probe_reasoning_results")
    parser.add_argument("--probes-dir", default="Results/phase_probe_results")
    parser.add_argument("--responses-dir", default="Results/probe_responses")
    parser.add_argument("--synthesized-dir", default="Results/redteam_synthesized")
    parser.add_argument("--judged-dir", default="Results/redteam_judged")
    parser.add_argument("--task-plan-model", default="gpt-4o-mini")
    parser.add_argument("--reasoning-model", default="o4-mini")
    parser.add_argument("--optimizer-model", default="gpt-4o-mini")
    parser.add_argument("--generator-model", default="gpt-4o-mini")
    parser.add_argument("--responder-model", default="gpt-4o-mini")
    parser.add_argument("--synthesizer-model", default="deepseek-v4-pro")
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--no-repair-skips", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(
        input_path=args.input,
        breakdown_dir=args.breakdown_dir,
        reasoning_dir=args.reasoning_dir,
        probes_dir=args.probes_dir,
        responses_dir=args.responses_dir,
        synthesized_dir=args.synthesized_dir,
        judged_dir=args.judged_dir,
        task_plan_model=args.task_plan_model,
        reasoning_model=args.reasoning_model,
        optimizer_model=args.optimizer_model,
        generator_model=args.generator_model,
        responder_model=args.responder_model,
        synthesizer_model=args.synthesizer_model,
        judge_model=args.judge_model,
        repair_skips=not args.no_repair_skips,
        verbose=True,
    )

    print("\nPipeline outputs:")
    for key, value in result.items():
        print(f"  {key}: {value}")
