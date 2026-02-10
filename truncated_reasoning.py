"""Utilities to truncate `<think>...</think>` reasoning at controlled lengths."""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _find_think_bounds(response: str) -> Optional[Tuple[int, int]]:
    """Return start/end offsets of the CoT content, or None when `<think>` is missing."""
    open_idx = response.find(THINK_OPEN)
    if open_idx == -1:
        return None

    start = open_idx + len(THINK_OPEN)
    close_idx = response.find(THINK_CLOSE, start)
    end = close_idx if close_idx != -1 else len(response)
    return start, end


def extract_thinking_process(response: str) -> str:
    """Extract CoT content between `<think>` and `</think>`."""
    bounds = _find_think_bounds(response)
    if bounds is None:
        return ""
    start, end = bounds
    return response[start:end]


def _clamp_cutoff(cot_length: int, requested: int) -> int:
    return max(0, min(requested, cot_length))


def force_answer(
    response: str,
    cot_cutoff_chars: int,
    forced_prefix: str = "\nFinal Answer: ",
) -> str:
    """
    Truncate CoT by character count and force answer section.

    If no `<think>` section exists, the response is returned unchanged.
    """
    bounds = _find_think_bounds(response)
    if bounds is None:
        return response

    start, end = bounds
    cot = response[start:end]
    cutoff = _clamp_cutoff(len(cot), cot_cutoff_chars)

    prefix = response[:start]
    truncated_cot = cot[:cutoff]
    return f"{prefix}{truncated_cot}{THINK_CLOSE}{forced_prefix}"


def truncate_by_percentage(
    response: str,
    percentage: float,
    forced_prefix: str = "\nFinal Answer: ",
) -> str:
    """Truncate CoT to `percentage` of original CoT character length."""
    cot = extract_thinking_process(response)
    cutoff = round(len(cot) * percentage)
    return force_answer(response, cot_cutoff_chars=cutoff, forced_prefix=forced_prefix)


def truncate_by_token_count(
    response: str,
    cot_cutoff_tokens: int,
    tokenizer,
    forced_prefix: str = "\nFinal Answer: ",
) -> str:
    """
    Truncate CoT by token count for stable cross-example control.

    Uses tokenizer encode/decode only on the CoT segment.
    """
    bounds = _find_think_bounds(response)
    if bounds is None:
        return response

    start, end = bounds
    cot = response[start:end]
    cot_token_ids = tokenizer.encode(cot, add_special_tokens=False)
    cutoff = _clamp_cutoff(len(cot_token_ids), cot_cutoff_tokens)
    truncated = tokenizer.decode(cot_token_ids[:cutoff], skip_special_tokens=True)
    return f"{response[:start]}{truncated}{THINK_CLOSE}{forced_prefix}"


def truncate_batch_by_percentage(
    responses: List[str],
    percentage: float,
    forced_prefix: str = "\nFinal Answer: ",
) -> Tuple[List[str], List[Dict[str, int]]]:
    """
    Batch truncation helper with per-sample metadata for debugging.

    Metadata keys:
    - has_think: 0/1
    - original_chars
    - cutoff_chars
    """
    outputs: List[str] = []
    metadata: List[Dict[str, int]] = []

    for response in responses:
        cot = extract_thinking_process(response)
        has_think = 1 if cot else 0
        cutoff = round(len(cot) * percentage)
        outputs.append(
            force_answer(response, cot_cutoff_chars=cutoff, forced_prefix=forced_prefix)
        )
        metadata.append(
            {
                "has_think": has_think,
                "original_chars": len(cot),
                "cutoff_chars": _clamp_cutoff(len(cot), cutoff),
            }
        )

    return outputs, metadata


def _pick_records_key(data: dict) -> str:
    if isinstance(data.get("results"), list):
        return "results"
    if isinstance(data.get("model_response"), list):
        return "model_response"
    raise ValueError("Expected top-level key 'results' or 'model_response' as a list.")


def _extract_response_text(value) -> str:
    """
    Robustly extract response text from str/list/dict payloads.
    """
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


def _parse_percentages(value: str) -> List[float]:
    percentages = [float(v.strip()) for v in value.split(",") if v.strip()]
    for pct in percentages:
        if pct < 0.0 or pct > 1.0:
            raise ValueError(f"Percentage must be in [0, 1], got {pct}")
    return percentages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create CoT-truncated variants from a results JSON file."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON (expects a top-level 'results' list).",
    )
    parser.add_argument(
        "--percentages",
        type=str,
        default="0.25,0.5,0.75,1.0",
        help="Comma-separated percentages in [0,1], e.g. '0.1,0.2,0.5'.",
    )
    parser.add_argument(
        "--response_key",
        type=str,
        default="model_response",
        help="Key containing the full model response.",
    )
    parser.add_argument(
        "--forced_prefix",
        type=str,
        default="\nFinal Answer: ",
        help="Text appended right after inserted </think>.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/gsm8k",
        help="Directory for truncated output files.",
    )
    args = parser.parse_args()

    percentages = _parse_percentages(args.percentages)

    with open(args.input_json, "r") as f:
        data = json.load(f)

    records_key = _pick_records_key(data)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input_json))[0]

    for pct in percentages:
        out_data = dict(data)
        out_results = []
        for row in data[records_key]:
            row_copy = dict(row)
            response = _extract_response_text(row_copy.get(args.response_key, ""))
            truncated = truncate_by_percentage(
                response=response,
                percentage=pct,
                forced_prefix=args.forced_prefix,
            )
            row_copy[f"{args.response_key}_truncated"] = truncated
            row_copy["truncation_percentage"] = pct
            row_copy["truncation_original_chars"] = len(
                extract_thinking_process(response)
            )
            out_results.append(row_copy)

        out_data[records_key] = out_results
        out_data["truncation_percentage"] = pct

        out_path = os.path.join(
            args.output_dir, f"{base_name}.truncated_{int(pct * 100):03d}.json"
        )
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
