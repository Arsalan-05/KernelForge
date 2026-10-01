"""Phase 3: baseline evaluation of the un-fine-tuned base model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.common import (
    aggregate_metrics,
    evaluate_generated_kernel,
    print_metrics_table,
    save_eval_report,
)
from kernelforge.dataset import interleave_by_category, load_training_data, make_split, template_of
from kernelforge.model import GenerationConfig, KernelGenerator, load_generation_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline eval: un-fine-tuned model on held-out test set.")
    parser.add_argument("--config", type=Path, default=Path("training/configs/qwen2.5-coder-1.5b.json"))
    parser.add_argument("--dataset", type=Path, default=None, help="Path to verified dataset.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of test examples")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--skip-verification", action="store_true", help="Parse only, no harness run")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model inference; use ground-truth kernels to validate the eval pipeline",
    )
    parser.add_argument("--split", choices=["template", "random"], default="template",
                        help="template = held-out ops (no leakage); random = shape-level split")
    args = parser.parse_args()

    entries = load_training_data(args.dataset)
    if not entries:
        print("No dataset entries found. Run data/build_dataset.py --device cuda first,", file=sys.stderr)
        print("or ensure data/examples/*.json exists for a minimal dry-run.", file=sys.stderr)
        sys.exit(1)

    _, _, test_entries = make_split(entries, args.split)
    if args.limit:
        test_entries = interleave_by_category(test_entries)[: args.limit]

    held_out = sorted({template_of(e) for e in test_entries})
    print(f"Evaluating {len(test_entries)} held-out test examples (baseline, no adapter); "
          f"{args.split} split, templates: {', '.join(held_out)}")

    gen_config = load_generation_config(args.config)
    generator = None if args.dry_run else KernelGenerator(gen_config)

    results = []
    for i, entry in enumerate(test_entries, 1):
        print(f"[{i}/{len(test_entries)}] {entry['id']} ({entry['category']})")

        if args.dry_run:
            generated = (
                f"kernel:\n{entry['triton_kernel']}\n\n"
                f"explanation:\n{entry['optimization_explanation']}"
            )
        else:
            generator.load()
            generated = generator.generate(entry["description"], entry["pytorch_reference"])

        result = evaluate_generated_kernel(
            entry,
            generated,
            device=args.device,
            warmup=args.warmup,
            iters=args.iters,
            timeout_s=args.timeout,
            skip_verification=args.skip_verification,
        )
        results.append(result)

        if result.verification:
            print(
                f"  status={result.verification.status} correct={result.verification.correct} "
                f"speedup={result.verification.speedup}"
            )
        elif result.error:
            print(f"  error: {result.error}")
        else:
            print(f"  parsed_kernel={result.parsed.has_kernel} has_explanation={result.has_explanation}")

    metrics = aggregate_metrics(results)
    print_metrics_table(metrics, title="Baseline (un-fine-tuned)")

    explanation_rate = sum(1 for r in results if r.has_explanation) / len(results) * 100
    print(f"\nExplanation field present in {explanation_rate:.1f}% of outputs.")

    report_path = save_eval_report(
        run_name="baseline",
        model_name=gen_config.model_name,
        adapter_path=None,
        device=args.device,
        results=results,
        metrics=metrics,
        extra={
            "dry_run": args.dry_run,
            "split": args.split,
            "held_out_templates": held_out,
            "skip_verification": args.skip_verification,
            "explanation_rate_pct": explanation_rate,
        },
    )
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
