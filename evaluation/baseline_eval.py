"""Phase 3: baseline evaluation of the un-fine-tuned base model.

    python evaluation/baseline_eval.py --device cuda --limit 25   # Kaggle T4
    python evaluation/baseline_eval.py --dry-run --device cpu     # pipeline check, no model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.common import add_eval_args, aggregate_metrics, print_metrics_table, print_outcomes, run_eval, save_eval_report
from kernelforge.dataset import load_eval_entries, make_split, template_of
from kernelforge.model import KernelGenerator, load_generation_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline eval: un-fine-tuned model on held-out test set.")
    add_eval_args(parser)
    args = parser.parse_args()

    entries, source = load_eval_entries(args.dataset, dry_run=args.dry_run)
    if not entries:
        print("No verified dataset entries found. Run data/build_dataset.py --device cuda first, "
              "or pass --dry-run to check the pipeline on the unverified template candidates.", file=sys.stderr)
        sys.exit(1)

    _, _, test_entries = make_split(entries, args.split)
    held_out = sorted({template_of(e) for e in test_entries})
    print(f"Data: {source}. {len(test_entries)} held-out test examples (baseline, no adapter); "
          f"{args.split} split, templates: {', '.join(held_out)}")
    if args.dry_run:
        print("DRY RUN: scoring the ground-truth kernels, not model output. Not a baseline result.")

    gen_config = load_generation_config(args.config)
    generator = None if args.dry_run else KernelGenerator(gen_config)
    results = run_eval(test_entries, args, generator)

    metrics = aggregate_metrics(results)
    print_metrics_table(metrics, title="DRY RUN (ground truth)" if args.dry_run else "Baseline (un-fine-tuned)")
    print_outcomes(results)
    explanation_rate = sum(1 for r in results if r.has_explanation) / len(results) * 100
    print(f"\nExplanation field present in {explanation_rate:.1f}% of outputs.")

    report_path = save_eval_report(
        # "dryrun_" prefix so a dry run is never picked up as the baseline to compare against.
        run_name="dryrun_baseline" if args.dry_run else "baseline",
        model_name=gen_config.model_name,
        adapter_path=None,
        device=args.device,
        results=results,
        metrics=metrics,
        extra={
            "dry_run": args.dry_run,
            "data_source": source,
            "split": args.split,
            "held_out_templates": held_out,
            "skip_verification": args.skip_verification,
            "interpret": args.interpret,
            "max_new_tokens": gen_config.max_new_tokens,
            "backend": generator.backend if generator else None,
            "explanation_rate_pct": explanation_rate,
        },
    )
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
