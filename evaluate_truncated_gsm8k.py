"""
Evaluate continuations from truncated CoT prompts on GSM8K-style result files.

This script is intentionally aligned with evaluate_gsm8k.py:
- model loading via utils.load_model_and_vectors(...)
- generation via model.generate(...) + model.generator.output.save()
- answer extraction via evaluate_gsm8k.extract_answer / extract_ground_truth

Input files are expected to be outputs from truncated_reasoning.py, typically
containing `model_response_truncated` in each row.

Output is a nested JSON to make cross-truncation comparison easier.
"""

import argparse
import gc
import glob
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# Reuse extraction logic from evaluate_gsm8k.py
from evaluate_gsm8k import extract_answer, extract_ground_truth

# Add utils to path (same pattern as evaluate_gsm8k.py)
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
import utils


def _pick_records_key(data: dict) -> str:
    if isinstance(data.get("results"), list):
        return "results"
    if isinstance(data.get("model_response"), list):
        return "model_response"
    raise ValueError("Expected top-level key 'results' or 'model_response' as a list.")


def _extract_response_text(value: Any) -> str:
    """Robust extraction of response text from str/list/dict payloads."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            text = _extract_response_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in (
            "model_response",
            "response",
            "text",
            "result",
            "generated_text",
            "output",
            "content",
            "prediction",
            "predictions",
        ):
            if key in value:
                text = _extract_response_text(value.get(key))
                if text:
                    return text
    return str(value)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _parse_csv_paths(value: str) -> List[Path]:
    if not value:
        return []
    out: List[Path] = []
    for chunk in value.split(","):
        candidate = chunk.strip()
        if candidate:
            out.append(Path(candidate))
    return out


def _parse_glob_patterns(value: str) -> List[str]:
    if not value:
        return []
    return [chunk.strip() for chunk in value.split(",") if chunk.strip()]


def _discover_input_files(input_jsons: str, input_glob: str) -> List[Path]:
    files: List[Path] = []
    files.extend(_parse_csv_paths(input_jsons))

    for pattern in _parse_glob_patterns(input_glob):
        for match in glob.glob(pattern):
            files.append(Path(match))

    deduped = sorted({path.resolve() for path in files})
    return deduped


def _extract_truncation_percentage(payload: dict, rows: List[dict], path: Path) -> Optional[float]:
    top_level = _safe_float(payload.get("truncation_percentage"))
    if top_level is not None:
        return top_level

    for row in rows:
        if isinstance(row, dict):
            row_pct = _safe_float(row.get("truncation_percentage"))
            if row_pct is not None:
                return row_pct

    filename_match = re.search(r"\.truncated_(\d{3})\.json$", path.name)
    if filename_match:
        return int(filename_match.group(1)) / 100.0

    return None


def _variant_label(path: Path, truncation_pct: Optional[float]) -> str:
    if truncation_pct is not None:
        return f"{int(round(truncation_pct * 100)):03d}"

    filename_match = re.search(r"\.truncated_(\d{3})\.json$", path.name)
    if filename_match:
        return filename_match.group(1)

    return path.stem


def _sort_key_for_variant(variant: dict) -> Tuple[float, str]:
    pct = variant.get("truncation_percentage")
    if isinstance(pct, (int, float)):
        return float(pct), variant["variant_label"]
    return 999.0, variant["variant_label"]


def _build_batched_input_ids(
    tokenizer,
    prompts: List[str],
) -> Tuple[torch.Tensor, List[int]]:
    encoded: List[List[int]] = []
    lengths: List[int] = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not token_ids:
            token_ids = tokenizer.encode(" ", add_special_tokens=False)
        encoded.append(token_ids)
        lengths.append(len(token_ids))

    max_len = max(lengths)
    batch_size = len(prompts)

    input_ids = torch.full(
        (batch_size, max_len),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device="cuda",
    )

    for idx, token_ids in enumerate(encoded):
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device="cuda")
        input_ids[idx, -len(token_ids) :] = token_tensor

    return input_ids, lengths


def _decode_batch_outputs(
    outputs: List[torch.Tensor],
    tokenizer,
    max_input_len: int,
) -> List[Dict[str, str]]:
    decoded: List[Dict[str, str]] = []
    for output in outputs:
        output_cpu = output.detach().cpu()

        if tokenizer.pad_token_id is not None:
            non_pad_idx = (output_cpu != tokenizer.pad_token_id).nonzero(as_tuple=False)
            start = int(non_pad_idx[0].item()) if len(non_pad_idx) > 0 else 0
        else:
            start = 0

        trimmed = output_cpu[start:]
        full_text = tokenizer.decode(trimmed, skip_special_tokens=False)

        generated_only = output_cpu[max_input_len:]
        generated_text = tokenizer.decode(generated_only, skip_special_tokens=False)

        decoded.append(
            {
                "full_text": full_text,
                "generated_text": generated_text,
            }
        )

    return decoded


def _prepare_rows(
    rows: List[dict],
    response_key: str,
    fallback_response_key: str,
    n_samples: int,
) -> List[dict]:
    selected_rows = rows
    if n_samples > 0:
        selected_rows = rows[: min(n_samples, len(rows))]

    prepared: List[dict] = []
    for row_idx, row in enumerate(selected_rows):
        if not isinstance(row, dict):
            continue

        prompt = _extract_response_text(row.get(response_key))
        prompt_source_key = response_key
        if not prompt:
            prompt = _extract_response_text(row.get(fallback_response_key))
            prompt_source_key = fallback_response_key

        ground_truth_answer = _safe_float(row.get("ground_truth_answer"))
        if ground_truth_answer is None:
            ground_truth_answer = extract_ground_truth(str(row.get("ground_truth_solution", "")))

        prepared.append(
            {
                "row_index": row_idx,
                "question": str(row.get("question", "")),
                "ground_truth_solution": str(row.get("ground_truth_solution", "")),
                "ground_truth_answer": ground_truth_answer,
                "prompt_text": prompt,
                "prompt_source_key": prompt_source_key,
                "baseline_model_response": _extract_response_text(row.get("model_response")),
                "baseline_predicted_answer": _safe_float(row.get("predicted_answer")),
                "baseline_is_correct": bool(row.get("is_correct"))
                if row.get("is_correct") is not None
                else None,
            }
        )
    return prepared


def _evaluate_variant(
    model,
    tokenizer,
    variant_meta: dict,
    prepared_rows: List[dict],
    batch_size: int,
    max_tokens: int,
    disable_tqdm: bool,
) -> Dict[str, Any]:
    rows_out: List[dict] = []
    correct = 0
    total = 0

    if not prepared_rows:
        return {"rows": rows_out, "correct": 0, "total": 0, "accuracy": 0.0}

    num_batches = (len(prepared_rows) + batch_size - 1) // batch_size

    for batch_idx in tqdm(
        range(num_batches),
        desc=f"Variant {variant_meta['variant_label']}",
        disable=disable_tqdm,
    ):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(prepared_rows))
        batch_rows = prepared_rows[start:end]
        prompts = [row["prompt_text"] for row in batch_rows]

        input_ids, prompt_token_lengths = _build_batched_input_ids(tokenizer, prompts)

        with model.generate(
            {
                "input_ids": input_ids,
                "attention_mask": (input_ids != tokenizer.pad_token_id).long(),
            },
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.pad_token_id,
        ) as tracer:
            outputs = model.generator.output.save()

        decoded_outputs = _decode_batch_outputs(
            outputs=outputs,
            tokenizer=tokenizer,
            max_input_len=input_ids.shape[1],
        )

        for row, prompt_len, decoded in zip(batch_rows, prompt_token_lengths, decoded_outputs):
            full_text = decoded["full_text"]
            generated_text = decoded["generated_text"]

            predicted_answer = extract_answer(full_text)
            ground_truth = row["ground_truth_answer"]

            is_correct = False
            if predicted_answer is not None and ground_truth is not None:
                is_correct = abs(predicted_answer - ground_truth) < 1e-3

            total += 1
            if is_correct:
                correct += 1

            rows_out.append(
                {
                    "row_index": row["row_index"],
                    "question": row["question"],
                    "ground_truth_solution": row["ground_truth_solution"],
                    "ground_truth_answer": ground_truth,
                    "prompt_source_key": row["prompt_source_key"],
                    "prompt_text": row["prompt_text"],
                    "prompt_chars": len(row["prompt_text"]),
                    "prompt_tokens": prompt_len,
                    "generated_continuation": generated_text,
                    "continued_response": full_text,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "baseline_model_response": row["baseline_model_response"],
                    "baseline_predicted_answer": row["baseline_predicted_answer"],
                    "baseline_is_correct": row["baseline_is_correct"],
                }
            )

        torch.cuda.empty_cache()
        gc.collect()

    accuracy = correct / total if total > 0 else 0.0
    return {"rows": rows_out, "correct": correct, "total": total, "accuracy": accuracy}


def _build_nested_comparison(variant_results: List[dict]) -> List[dict]:
    rows_by_index: Dict[int, dict] = {}

    for variant in variant_results:
        variant_label = variant["variant_label"]
        truncation_pct = variant.get("truncation_percentage")
        source_file = variant["source_file"]
        for row in variant["rows"]:
            row_idx = row["row_index"]

            if row_idx not in rows_by_index:
                rows_by_index[row_idx] = {
                    "row_index": row_idx,
                    "question": row["question"],
                    "ground_truth_solution": row["ground_truth_solution"],
                    "ground_truth_answer": row["ground_truth_answer"],
                    "baseline": {
                        "model_response": row.get("baseline_model_response", ""),
                        "predicted_answer": row.get("baseline_predicted_answer"),
                        "is_correct": row.get("baseline_is_correct"),
                    },
                    "variants": {},
                }

            rows_by_index[row_idx]["variants"][variant_label] = {
                "truncation_percentage": truncation_pct,
                "source_file": source_file,
                "prompt_source_key": row["prompt_source_key"],
                "prompt_text": row["prompt_text"],
                "prompt_chars": row["prompt_chars"],
                "prompt_tokens": row["prompt_tokens"],
                "generated_continuation": row["generated_continuation"],
                "continued_response": row["continued_response"],
                "predicted_answer": row["predicted_answer"],
                "is_correct": row["is_correct"],
            }

    return [rows_by_index[idx] for idx in sorted(rows_by_index.keys())]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate continued answers from truncated CoT prompts using the "
            "evaluate_gsm8k.py-style inference path, then build nested comparison JSON."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Model identifier/path (same style as evaluate_gsm8k.py).",
    )
    parser.add_argument(
        "--input_jsons",
        type=str,
        default="",
        help="Comma-separated truncated JSON paths.",
    )
    parser.add_argument(
        "--input_glob",
        type=str,
        default="",
        help="Comma-separated glob patterns for truncated JSON paths.",
    )
    parser.add_argument(
        "--response_key",
        type=str,
        default="model_response_truncated",
        help="Key used as continuation prompt in each row.",
    )
    parser.add_argument(
        "--fallback_response_key",
        type=str,
        default="model_response",
        help="Fallback key if --response_key is missing/empty.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=0,
        help="Number of rows per file to evaluate (0 = all).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for continuation generation.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum number of newly generated tokens per continuation.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        default=False,
        help="Load model in 8-bit mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Output path for nested comparison JSON (auto if omitted).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/gsm8k",
        help="Directory for auto-generated output path.",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        default=False,
        help="Disable progress bars.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_files = _discover_input_files(args.input_jsons, args.input_glob)
    if not input_files:
        raise ValueError(
            "No input files found. Provide --input_jsons and/or --input_glob."
        )

    print("=" * 80)
    print("Evaluate Truncated GSM8K Continuations")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Input files: {len(input_files)}")
    for path in input_files:
        print(f"  - {path}")
    print(f"Response key: {args.response_key}")
    print(f"Fallback key: {args.fallback_response_key}")
    print(f"N samples per file: {args.n_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Load in 8-bit: {args.load_in_8bit}")
    print("=" * 80)

    print(f"\nLoading model {args.model}...")
    model, tokenizer, _ = utils.load_model_and_vectors(
        compute_features=False,
        model_name=args.model,
        load_in_8bit=args.load_in_8bit,
    )
    print("Model loaded successfully!")

    variant_inputs: List[dict] = []
    for path in input_files:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        records_key = _pick_records_key(payload)
        rows = payload[records_key]
        trunc_pct = _extract_truncation_percentage(payload, rows, path)
        label = _variant_label(path, trunc_pct)

        prepared_rows = _prepare_rows(
            rows=rows,
            response_key=args.response_key,
            fallback_response_key=args.fallback_response_key,
            n_samples=args.n_samples,
        )

        variant_inputs.append(
            {
                "path": str(path),
                "records_key": records_key,
                "truncation_percentage": trunc_pct,
                "variant_label": label,
                "prepared_rows": prepared_rows,
            }
        )

    variant_inputs = sorted(variant_inputs, key=_sort_key_for_variant)

    variant_results: List[dict] = []
    for variant in variant_inputs:
        print(
            f"\nRunning variant {variant['variant_label']} "
            f"(truncation={variant['truncation_percentage']})"
        )
        evaluation = _evaluate_variant(
            model=model,
            tokenizer=tokenizer,
            variant_meta=variant,
            prepared_rows=variant["prepared_rows"],
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            disable_tqdm=args.disable_tqdm,
        )

        print(
            f"Variant {variant['variant_label']} accuracy: "
            f"{evaluation['accuracy']:.2%} ({evaluation['correct']}/{evaluation['total']})"
        )

        variant_results.append(
            {
                "variant_label": variant["variant_label"],
                "truncation_percentage": variant["truncation_percentage"],
                "source_file": variant["path"],
                "records_key": variant["records_key"],
                "n_samples": len(evaluation["rows"]),
                "correct": evaluation["correct"],
                "total": evaluation["total"],
                "accuracy": evaluation["accuracy"],
                "rows": evaluation["rows"],
            }
        )

    nested_comparison = _build_nested_comparison(variant_results)

    per_variant_metrics = [
        {
            "variant_label": variant["variant_label"],
            "truncation_percentage": variant["truncation_percentage"],
            "source_file": variant["source_file"],
            "n_samples": variant["n_samples"],
            "correct": variant["correct"],
            "total": variant["total"],
            "accuracy": variant["accuracy"],
        }
        for variant in variant_results
    ]

    if args.output_json:
        output_path = Path(args.output_json)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        model_id = args.model.split("/")[-1].lower()
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output_dir) / f"gsm8k_{model_id}_{date}.truncated_compare.json"

    output_payload = {
        "model": args.model,
        "created_at": datetime.now().isoformat(),
        "settings": {
            "response_key": args.response_key,
            "fallback_response_key": args.fallback_response_key,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "load_in_8bit": args.load_in_8bit,
            "seed": args.seed,
        },
        "input_files": [str(path) for path in input_files],
        "variant_metrics": per_variant_metrics,
        "comparison": nested_comparison,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    print("\n" + "=" * 80)
    print(f"Saved nested comparison: {output_path}")
    print(f"Rows in comparison: {len(nested_comparison)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
