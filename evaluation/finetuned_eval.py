"""Phase 5: evaluate the fine-tuned model on the held-out test set.

    python evaluation/finetuned_eval.py --device cuda --checkpoint-run training/checkpoints/run_X
    python evaluation/finetuned_eval.py --device cuda --checkpoint-run training/checkpoints/run_X \
        --num-candidates 4 --repair-rounds 1        # + best-of-4 + 1 repair on 25 prompts

See evaluation/common.py for --resume, --max-minutes and the search flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.common import EVAL_REPETITION_PENALTY, RESULTS_DIR, add_eval_args, evaluate
from kernelforge.dataset import load_eval_entries, make_split, template_of
from kernelforge.model import KernelGenerator, load_generation_config


def load_baseline_report(results_dir: Path, test_ids: list[str]) -> tuple[dict | None, str | None]:
    """Most recent real zero-shot baseline report that scored exactly the same test entries.

    Dry runs are skipped (they score ground truth), few-shot baselines too (the
    fine-tuned model is prompted zero-shot), and so is any report on a different
    set of entries, since per-category numbers would not be comparable.
    """
    for path in sorted(results_dir.glob("baseline_*.json"), reverse=True):
        report = json.loads(path.read_text())
        if report.get("dry_run") or report.get("prompt_mode", "zero_shot") != "zero_shot":
            continue
        if sorted(r["id"] for r in report.get("results", [])) != sorted(test_ids):
            print(f"Skipping {path.name} for comparison: it scored a different set of test entries.")
            continue
        return report.get("per_category"), path.name
    return None, None


def _fmt(value, spec: str, suffix: str) -> str:
    return f"{value:{spec}}{suffix}" if value is not None else "n/a"


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on held-out test set.")
    add_eval_args(parser)
    parser.add_argument("--adapter", type=Path, default=None,
                        help="LoRA adapter directory (default: <checkpoint-run>/adapter)")
    parser.add_argument("--checkpoint-run", type=Path, default=None, help="Training run dir with held_out_test.jsonl")
    args = parser.parse_args(argv)

    if args.checkpoint_run and (args.checkpoint_run / "held_out_test.jsonl").exists():
        test_entries = [
            json.loads(line)
            for line in (args.checkpoint_run / "held_out_test.jsonl").read_text().splitlines()
            if line.strip()
        ]
        source = f"{args.checkpoint_run / 'held_out_test.jsonl'}"
    else:
        entries, source = load_eval_entries(args.dataset, dry_run=args.dry_run)
        if not entries:
            print("No verified dataset entries found.", file=sys.stderr)
            sys.exit(1)
        _, _, test_entries = make_split(entries, args.split)

    gen_config = load_generation_config(args.config)
    gen_config.max_new_tokens = args.max_new_tokens
    gen_config.repetition_penalty = EVAL_REPETITION_PENALTY
    adapter = args.adapter
    if adapter is None and args.checkpoint_run and (args.checkpoint_run / "adapter").exists():
        adapter = args.checkpoint_run / "adapter"
    if adapter is None and not args.dry_run:
        # Without an adapter this would silently evaluate the base model under a "finetuned" label.
        print("No adapter: pass --adapter or a --checkpoint-run that contains adapter/.", file=sys.stderr)
        sys.exit(1)
    gen_config.adapter_path = str(adapter) if adapter else None

    held_out = sorted({template_of(e) for e in test_entries})
    print(f"Data: {source}. {len(test_entries)} held-out test examples (fine-tuned, adapter={gen_config.adapter_path}); "
          f"templates: {', '.join(held_out)}")
    if args.dry_run:
        print("DRY RUN: scoring the ground-truth kernels, not model output. Not a fine-tuned result.")

    generator = None if args.dry_run else KernelGenerator(gen_config)
    # Adapters from `finetune.py --dry-run` (tiny random model) live in dryrun_* dirs.
    dry = args.dry_run or "dryrun_" in str(gen_config.adapter_path or "")
    path, report = evaluate(
        args, test_entries, run_name="dryrun_finetuned" if dry else "finetuned", gen_config=gen_config,
        generator=generator, prompt_mode="zero_shot",
        extra={"data_source": source, "held_out_templates": held_out, "dry_run_adapter": dry and not args.dry_run},
    )

    baseline, baseline_file = load_baseline_report(RESULTS_DIR, [r["id"] for r in report["results"]])
    report.update(baseline_report=baseline_file, baseline_per_category=baseline)
    path.write_text(json.dumps(report, indent=2))
    if baseline:
        print(f"\n=== Baseline ({baseline_file}) vs Fine-tuned, pass@1 per category ===")
        print(f"{'Category':<20} {'Base correct':>13} {'FT correct':>11} {'Base speedup':>13} {'FT speedup':>11}")
        for category, ft in report["per_category"].items():
            base = baseline.get(category, {})
            print(f"{category:<20} {_fmt(base.get('pct_correct'), '.1f', '%'):>13} "
                  f"{_fmt(ft['pct_correct'], '.1f', '%'):>11} {_fmt(base.get('avg_speedup'), '.2f', 'x'):>13} "
                  f"{_fmt(ft['avg_speedup'], '.2f', 'x'):>11}")
    else:
        print("\nNo comparable baseline report found (zero-shot, same test entries, not a dry run).")
    return path


if __name__ == "__main__":
    main()
