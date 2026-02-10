"""
GSM8K Evaluation Script for DeepSeek-R1-Distill Models

This script evaluates reasoning models on the GSM8K mathematical reasoning benchmark.
It supports both standard and steered generation modes.
"""

import argparse
import gc
import json
import os
import re
import sys
from datetime import datetime

import torch
from datasets import load_dataset
from tqdm import tqdm

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
import utils


def _parse_numeric_candidate(text):
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


def extract_answer(text):
    """
    Extract the numerical answer from model output.
    Looks for patterns like "####" (GSM8K format) or boxed answers.
    """
    # First try to extract from <think>...</think> tags and find answer after
    if "</think>" in text:
        # Get text after thinking
        answer_section = text.split("</think>")[-1]
    else:
        answer_section = text

    # Look for common answer patterns
    patterns = [
        r"\\boxed\{([^}]*)\}",  # LaTeX boxed content
        r"####\s*([^\n]+)",  # GSM8K format
        r"final answer[:\s]*([^\n]+)",  # "Final Answer: ..."
        r"the answer is[:\s]*([^\n]+)",  # "the answer is X"
        r"answer[:\s]*([^\n]+)",  # "answer: X"
        r"=\s*([^\n]+)\s*$",  # Ends with "= X"
    ]

    for pattern in patterns:
        match = re.search(pattern, answer_section, re.IGNORECASE)
        if match:
            parsed = _parse_numeric_candidate(match.group(1))
            if parsed is not None:
                return parsed

    # Fallback: extract last number in the answer section
    numbers = re.findall(r"(-?\d[\d,]*(?:\.\d+)?)", answer_section)
    if numbers:
        parsed = _parse_numeric_candidate(numbers[-1])
        if parsed is not None:
            return parsed

    return None


def extract_ground_truth(answer_text):
    """Extract numerical answer from GSM8K ground truth format."""
    # GSM8K answers are in format: "#### 42"
    match = re.search(r"####\s*([^\n]+)", answer_text)
    if match:
        return _parse_numeric_candidate(match.group(1))
    return None


def evaluate_gsm8k(args):
    """Main evaluation function."""
    print("=" * 80)
    print("GSM8K Evaluation")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Load in 8-bit: {args.load_in_8bit}")
    print("=" * 80)

    # Load model
    print(f"\nLoading model {args.model}...")
    model, tokenizer, _ = utils.load_model_and_vectors(
        compute_features=False, model_name=args.model, load_in_8bit=args.load_in_8bit
    )
    print("Model loaded successfully!")

    # Load GSM8K dataset
    print("\nLoading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    # Limit number of samples
    if args.n_samples > 0:
        dataset = dataset.select(range(min(args.n_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} examples")

    # Prepare results storage
    results = []
    correct = 0
    total = 0

    # Create output directory
    os.makedirs("results/gsm8k", exist_ok=True)

    # Process in batches
    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

    for batch_idx in tqdm(range(num_batches), desc="Evaluating GSM8K"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(dataset))

        batch = dataset.select(range(start_idx, end_idx))

        # Prepare messages for batch
        messages_batch = []
        for example in batch:
            message = {"role": "user", "content": example["question"]}
            messages_batch.append(message)

        # Get batched input IDs
        max_token_length = max(
            [
                len(
                    tokenizer.apply_chat_template(
                        [msg], add_generation_prompt=True, return_tensors="pt"
                    )[0]
                )
                for msg in messages_batch
            ]
        )

        input_ids = torch.cat(
            [
                tokenizer.apply_chat_template(
                    [msg],
                    add_generation_prompt=True,
                    padding="max_length",
                    max_length=max_token_length,
                    return_tensors="pt",
                ).to("cuda")
                for msg in messages_batch
            ]
        )

        # Generate responses
        with model.generate(
            {
                "input_ids": input_ids,
                "attention_mask": (input_ids != tokenizer.pad_token_id).long(),
            },
            max_new_tokens=args.max_tokens,
            pad_token_id=tokenizer.pad_token_id,
        ) as tracer:
            outputs = model.generator.output.save()

        # Decode outputs
        responses = [
            tokenizer.decode(output, skip_special_tokens=False) for output in outputs
        ]

        # Evaluate each response
        for idx, (example, response) in enumerate(zip(batch, responses)):
            # Extract answers
            predicted_answer = extract_answer(response)
            ground_truth = extract_ground_truth(example["answer"])

            # Check if correct
            is_correct = False
            if predicted_answer is not None and ground_truth is not None:
                # Allow small floating point errors
                is_correct = abs(predicted_answer - ground_truth) < 1e-3

            # Update counters
            total += 1
            if is_correct:
                correct += 1

            # Store result
            result = {
                "question": example["question"],
                "ground_truth_solution": example["answer"],
                "ground_truth_answer": ground_truth,
                "model_response": response,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
            }
            results.append(result)

        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()

        # Periodic save
        if (batch_idx + 1) % args.save_every == 0 or batch_idx == num_batches - 1:
            accuracy = correct / total if total > 0 else 0
            print(
                f"\nBatch {batch_idx + 1}/{num_batches} - Accuracy: {accuracy:.2%} ({correct}/{total})"
            )

            # Save intermediate results
            model_id = args.model.split("/")[-1].lower()
            results_path = f"results/gsm8k/gsm8k_{model_id}.json"

            with open(results_path, "w") as f:
                json.dump(
                    {
                        "model": args.model,
                        "n_samples": len(results),
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": total,
                        "results": results,
                    },
                    f,
                    indent=2,
                )

    # Final results
    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Total examples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print("=" * 80)

    # Save final results
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = args.model.split("/")[-1].lower()
    results_path = f"results/gsm8k/gsm8k_{model_id}_{date}.json"

    with open(results_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "n_samples": total,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {results_path}")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on GSM8K")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (0 = all)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for processing"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2048, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        default=False,
        help="Load model in 8-bit mode",
    )
    parser.add_argument(
        "--save_every", type=int, default=5, help="Save results every N batches"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Run evaluation
    evaluate_gsm8k(args)
