"""
GSM8K evaluation via LightEval, with the same extraction logic as evaluate_gsm8k.py.

This script runs LightEval on GSM8K, reads per-sample details, and re-computes
accuracy using custom extraction patterns so you can validate them against
LightEval outputs.
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import (
    TransformersModelConfig,
)
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


def _parse_numeric_candidate(text: str | None) -> float | None:
    """Normalize common LaTeX/currency formatting and parse a number."""
    if text is None:
        return None

    cleaned = str(text).strip()
    cleaned = cleaned.replace("\\$", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("\\!", "")
    cleaned = cleaned.replace("\\,", "")
    cleaned = cleaned.replace(",", "")

    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def extract_answer(text: str) -> float | None:
    """
    Extract numerical answer from model output.
    Mirrors evaluate_gsm8k.py extraction behavior.
    """
    if "</think>" in text:
        answer_section = text.split("</think>")[-1]
    else:
        answer_section = text

    patterns = [
        r"\\boxed\{([^}]*)\}",
        r"####\s*([^\n]+)",
        r"final answer[:\s]*([^\n]+)",
        r"the answer is[:\s]*([^\n]+)",
        r"answer[:\s]*([^\n]+)",
        r"=\s*([^\n]+)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, answer_section, re.IGNORECASE)
        if match:
            parsed = _parse_numeric_candidate(match.group(1))
            if parsed is not None:
                return parsed

    numbers = re.findall(r"(-?\d[\d,]*(?:\.\d+)?)", answer_section)
    if numbers:
        parsed = _parse_numeric_candidate(numbers[-1])
        if parsed is not None:
            return parsed

    return None


def extract_ground_truth(answer_text: str) -> float | None:
    """Extract numerical answer from GSM8K-style '#### 42' solutions."""
    match = re.search(r"####\s*([^\n]+)", answer_text)
    if match:
        return _parse_numeric_candidate(match.group(1))
    return None


def _flatten_strings(value: Any) -> list[str]:
    out: list[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        s = value.strip()
        if s:
            out.append(s)
        return out
    if isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_flatten_strings(item))
        return out
    if isinstance(value, dict):
        for item in value.values():
            out.extend(_flatten_strings(item))
        return out
    return out


def _first_non_empty(*values: Any) -> str:
    for value in values:
        candidates = _flatten_strings(value)
        if candidates:
            return candidates[0]
    return ""


def _maybe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _extract_question(doc: dict[str, Any]) -> str:
    return _first_non_empty(
        doc.get("question"),
        doc.get("query"),
        doc.get("instruction"),
        doc.get("problem"),
        doc.get("prompt"),
        doc.get("ctx"),
    )


def _extract_ground_truth_solution(doc: dict[str, Any]) -> str:
    choices = doc.get("choices")
    gold_index = doc.get("gold_index")
    if isinstance(choices, list) and isinstance(gold_index, int):
        if 0 <= gold_index < len(choices):
            return _first_non_empty(choices[gold_index])

    return _first_non_empty(
        doc.get("answer"),
        doc.get("gold"),
        doc.get("target"),
        doc.get("reference"),
        doc.get("solution"),
    )


def _extract_model_response(model_response_obj: Any) -> str:
    if isinstance(model_response_obj, list) and model_response_obj:
        model_response_obj = model_response_obj[0]

    d = _maybe_dict(model_response_obj)

    preferred = [
        d.get("result"),
        d.get("text"),
        d.get("prediction"),
        d.get("predictions"),
        d.get("generated_text"),
        d.get("output"),
        d.get("response"),
        d.get("responses"),
    ]
    candidate = _first_non_empty(*preferred)
    if candidate:
        return candidate

    return _first_non_empty(model_response_obj)


def _find_latest_details_parquet(output_dir: str, task_hint: str) -> Path:
    task_slug = task_hint.replace("|", "_")
    output_path = Path(output_dir)
    search_roots = [
        output_path,
        output_path / "details",
        Path("results"),
        Path("."),
    ]

    patterns = [
        f"**/details_{task_slug}_*.parquet",
        "**/details_*gsm8k*.parquet",
        "**/details_*.parquet",
    ]

    candidates: list[Path] = []
    seen: set[Path] = set()

    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for match in root.glob(pattern):
                if match.is_file():
                    resolved = match.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        candidates.append(resolved)

    if not candidates:
        searched = ", ".join(str(root) for root in search_roots)
        raise FileNotFoundError(
            "No LightEval details parquet found. "
            f"Searched roots: {searched}. "
            "Pass --details_file explicitly if your LightEval run wrote to a custom path."
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def _safe_lighteval_metric_blob(value: Any) -> Any:
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolve_shared_model_path(model_name: str) -> str:
    """
    Prefer shared model storage when available, to mirror evaluate_gsm8k.py behavior.
    """
    if not model_name or model_name.startswith("/"):
        return model_name
    if "/" not in model_name:
        return model_name

    shared_path = f"/Brain/public/models/{model_name}"
    if os.path.exists(shared_path):
        print(f"Using shared model path: {shared_path}")
        return shared_path
    return model_name


def _build_model_config(args: argparse.Namespace) -> TransformersModelConfig:
    """
    Build a TransformersModelConfig compatible with multiple LightEval versions.
    """
    fields = getattr(TransformersModelConfig, "model_fields", {}) or {}
    field_names = set(fields.keys())

    model_value = _resolve_shared_model_path(args.model)
    kwargs: dict[str, Any] = {}

    for key in ("model_name", "pretrained"):
        if key in field_names:
            kwargs[key] = model_value
            break

    for key in ("batch_size",):
        if key in field_names:
            kwargs[key] = args.batch_size

    for key in ("dtype", "torch_dtype"):
        if key in field_names:
            kwargs[key] = args.dtype
            break

    for key in ("use_chat_template", "chat_template"):
        if key in field_names:
            kwargs[key] = args.use_chat_template
            break

    max_tokens_set = False
    for key in ("max_gen_toks", "max_new_tokens", "max_tokens"):
        if key in field_names:
            kwargs[key] = args.max_tokens
            max_tokens_set = True
            break

    if not max_tokens_set and "generation_parameters" in field_names:
        gp_candidates: list[Any] = [
            {"max_new_tokens": args.max_tokens},
            {"max_gen_toks": args.max_tokens},
            {"max_tokens": args.max_tokens},
        ]
        try:
            from lighteval.models.model_input import GenerationParameters

            gp_candidates = [
                GenerationParameters(max_new_tokens=args.max_tokens),
                GenerationParameters(max_gen_toks=args.max_tokens),
                GenerationParameters(max_tokens=args.max_tokens),
            ] + gp_candidates
        except Exception:
            pass

        for candidate in gp_candidates:
            try:
                test_kwargs = dict(kwargs)
                test_kwargs["generation_parameters"] = candidate
                return TransformersModelConfig(**test_kwargs)
            except Exception:
                continue

        print(
            "Warning: could not set max tokens on this LightEval version; "
            "using its default generation length."
        )

    return TransformersModelConfig(**kwargs)


def run_lighteval(args: argparse.Namespace) -> Any:
    print("=" * 80)
    print("GSM8K Evaluation (LightEval)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)

    tracker = EvaluationTracker(
        output_dir=args.output_dir,
        save_details=True,
        push_to_hub=False,
    )

    max_samples = None if args.n_samples == 0 else args.n_samples
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=max_samples,
    )

    model_config = _build_model_config(args)

    pipeline = Pipeline(
        tasks=args.task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=tracker,
        model_config=model_config,
    )

    print("\nRunning LightEval...")
    raw_results = pipeline.evaluate()

    if hasattr(pipeline, "save_and_push_results"):
        pipeline.save_and_push_results()
    if hasattr(pipeline, "show_results"):
        pipeline.show_results()

    return raw_results


def build_custom_results(details_parquet: Path, model_name: str) -> dict[str, Any]:
    details = load_dataset("parquet", data_files=str(details_parquet), split="train")

    results = []
    correct = 0
    total = 0

    for row in details:
        doc = _maybe_dict(row.get("__doc__"))
        model_resp_obj = row.get("__model_response__")

        question = _extract_question(doc)
        ground_truth_solution = _extract_ground_truth_solution(doc)
        model_response = _extract_model_response(model_resp_obj)

        predicted_answer = extract_answer(model_response)
        ground_truth_answer = extract_ground_truth(ground_truth_solution)

        is_correct = False
        if predicted_answer is not None and ground_truth_answer is not None:
            is_correct = abs(predicted_answer - ground_truth_answer) < 1e-3

        total += 1
        if is_correct:
            correct += 1

        results.append(
            {
                "question": question,
                "ground_truth_solution": ground_truth_solution,
                "ground_truth_answer": ground_truth_answer,
                "model_response": model_response,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
            }
        )

    accuracy = correct / total if total > 0 else 0.0
    return {
        "model": model_name,
        "n_samples": total,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def save_results(payload: dict[str, Any], model_name: str) -> tuple[Path, Path]:
    os.makedirs("results/gsm8k", exist_ok=True)
    model_id = model_name.split("/")[-1].lower()
    latest_path = Path(f"results/gsm8k/gsm8k_lighteval_{model_id}.json")

    with latest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = Path(f"results/gsm8k/gsm8k_lighteval_{model_id}_{date}.json")
    with timestamped_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return latest_path, timestamped_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GSM8K with LightEval and validate custom extraction patterns."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="lighteval|gsm8k|0|0",
        help="LightEval task string",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (0 = all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size passed to LightEval model config",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max generation tokens passed to LightEval model config",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Model dtype for LightEval Transformers backend",
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        default=False,
        help="Enable chat template usage in LightEval model config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/lighteval",
        help="LightEval output dir (results + details parquet)",
    )
    parser.add_argument(
        "--details_file",
        type=str,
        default="",
        help="Optional path to an existing LightEval details parquet file",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        default=False,
        help="Skip running LightEval and only process --details_file/latest details parquet",
    )

    args = parser.parse_args()

    raw_lighteval_results = None
    if not args.skip_eval:
        raw_lighteval_results = run_lighteval(args)

    if args.details_file:
        details_parquet = Path(args.details_file)
    else:
        details_parquet = _find_latest_details_parquet(args.output_dir, args.task)

    print(f"\nUsing details file: {details_parquet}")
    payload = build_custom_results(
        details_parquet=details_parquet, model_name=args.model
    )

    if raw_lighteval_results is not None:
        payload["lighteval_results"] = _safe_lighteval_metric_blob(
            raw_lighteval_results
        )

    latest_path, timestamped_path = save_results(payload=payload, model_name=args.model)

    print("\n" + "=" * 80)
    print("FINAL RESULTS (custom extraction over LightEval generations)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Total examples: {payload['total']}")
    print(f"Correct: {payload['correct']}")
    print(f"Accuracy: {payload['accuracy']:.2%}")
    print("=" * 80)
    print(f"Saved latest: {latest_path}")
    print(f"Saved timestamped: {timestamped_path}")


if __name__ == "__main__":
    main()
