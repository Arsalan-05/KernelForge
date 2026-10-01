"""Phase 5: evaluate the fine-tuned model on the held-out test set."""

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
from kernelforge.dataset import interleave_by_category, load_training_data, make_split
from kernelforge.model import KernelGenerator, load_generation_config


def _load_baseline_metrics(results_dir: Path) -> dict | None:
    """Load the most recent baseline report for side-by-side comparison."""
    reports = sorted(results_dir.glob("baseline_*.json"), reverse=True)
    if not reports:
        return None
    return json.loads(reports[0].read_text()).get("per_category")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on held-out test set.")
    parser.add_argument("--config", type=Path, default=Path("training/configs/qwen2.5-coder-1.5b.json"))
    parser.add_argument("--adapter", type=Path, required=False, help="Path to LoRA adapter directory")
    parser.add_argument("--checkpoint-run", type=Path, default=None, help="Training run dir with held_out_test.jsonl")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split", choices=["template", "random"], default="template",
                        help="template = held-out ops (no leakage); random = shape-level split")
    args = parser.parse_args()

    if args.checkpoint_run and (args.checkpoint_run / "held_out_test.jsonl").exists():
        test_entries = [
            json.loads(line)
            for line in (args.checkpoint_run / "held_out_test.jsonl").read_text().splitlines()
            if line.strip()
        ]
    else:
        entries = load_training_data(args.dataset)
        if not entries:
            print("No dataset entries found.", file=sys.stderr)
            sys.exit(1)
        _, _, test_entries = make_split(entries, args.split)

    if args.limit:
        test_entries = interleave_by_category(test_entries)[: args.limit]

    gen_config = load_generation_config(args.config)
    if args.adapter:
        gen_config.adapter_path = str(args.adapter)

    print(f"Evaluating {len(test_entries)} held-out test examples (fine-tuned).")
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

    metrics = aggregate_metrics(results)
    print_metrics_table(metrics, title="Fine-tuned")

    baseline = _load_baseline_metrics(Path(__file__).parent / "results")
    if baseline:
        print("\n=== Baseline vs Fine-tuned (per category) ===")
        print(f"{'Category':<20} {'Base correct':>13} {'FT correct':>11} {'Base speedup':>13} {'FT speedup':>11}")
        for category in metrics:
            ft = metrics[category].to_dict()
            base = baseline.get(category, {})
            base_correct = f"{base.get('pct_correct', 0):.1f}%" if base else "n/a"
            ft_correct = f"{ft.get('pct_correct', 0):.1f}%" if ft.get("pct_correct") is not None else "n/a"
            base_speedup = f"{base.get('avg_speedup', 0):.2f}x" if base.get("avg_speedup") else "n/a"
            ft_speedup = f"{ft.get('avg_speedup', 0):.2f}x" if ft.get("avg_speedup") else "n/a"
            print(f"{category:<20} {base_correct:>13} {ft_correct:>11} {base_speedup:>13} {ft_speedup:>11}")

    explanation_rate = sum(1 for r in results if r.has_explanation) / len(results) * 100
    print(f"\nExplanation field present in {explanation_rate:.1f}% of outputs.")

    report_path = save_eval_report(
        run_name="finetuned",
        model_name=gen_config.model_name,
        adapter_path=gen_config.adapter_path,
        device=args.device,
        results=results,
        metrics=metrics,
        extra={
            "dry_run": args.dry_run,
            "explanation_rate_pct": explanation_rate,
            "baseline_per_category": baseline,
        },
    )
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
