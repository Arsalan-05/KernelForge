"""Phase 3: baseline evaluation of the un-fine-tuned base model.

    python evaluation/baseline_eval.py --device cuda                       # greedy pass@1, all held-out prompts
    python evaluation/baseline_eval.py --device cuda --few-shot            # + one solved TRAIN example in the prompt
    python evaluation/baseline_eval.py --device cuda --num-candidates 4 --repair-rounds 1   # + search on 25 prompts
    python evaluation/baseline_eval.py --dry-run --device cpu              # pipeline check, no model

See evaluation/common.py for --resume, --max-minutes and the search flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.common import EVAL_REPETITION_PENALTY, add_eval_args, evaluate
from kernelforge.dataset import few_shot_picker, load_eval_entries, make_split, template_of
from kernelforge.model import KernelGenerator, load_generation_config


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Baseline eval: un-fine-tuned model on held-out test set.")
    add_eval_args(parser)
    parser.add_argument("--few-shot", action="store_true",
                        help="Show one solved example from the TRAIN split before each prompt")
    parser.add_argument("--few-shot-source", choices=["other_category", "same_category"], default="other_category",
                        help="other_category (default): the example is a different op family; "
                             "same_category: a sibling template of the same family (a much stronger hint)")
    args = parser.parse_args(argv)

    entries, source = load_eval_entries(args.dataset, dry_run=args.dry_run)
    if not entries:
        print("No verified dataset entries found. Run data/build_dataset.py --device cuda first, "
              "or pass --dry-run to check the pipeline on the unverified template candidates.", file=sys.stderr)
        sys.exit(1)

    train_entries, _, test_entries = make_split(entries, args.split)
    prompt_mode = "few_shot" if args.few_shot else "zero_shot"
    few_shot_source = args.few_shot_source if args.few_shot else None
    example_for = few_shot_picker(train_entries, args.few_shot_source) if args.few_shot else None
    held_out = sorted({template_of(e) for e in test_entries})
    print(f"Data: {source}. {len(test_entries)} held-out test examples (baseline, no adapter, {prompt_mode}); "
          f"{args.split} split, templates: {', '.join(held_out)}")
    if args.dry_run:
        print("DRY RUN: scoring the ground-truth kernels, not model output. Not a baseline result.")

    gen_config = load_generation_config(args.config)
    gen_config.max_new_tokens = args.max_new_tokens
    gen_config.repetition_penalty = EVAL_REPETITION_PENALTY
    generator = None if args.dry_run else KernelGenerator(gen_config)

    # "dryrun_" prefix so a dry run is never picked up as the baseline to compare against;
    # each few-shot source gets its own name (and results log) so modes never mix.
    run_name = ("dryrun_" if args.dry_run else "") + (f"baseline_few_shot_{args.few_shot_source}"
                                                      if args.few_shot else "baseline")
    path, _ = evaluate(
        args, test_entries, run_name=run_name, gen_config=gen_config, generator=generator,
        prompt_mode=prompt_mode, few_shot_source=few_shot_source, example_for=example_for,
        extra={"data_source": source, "held_out_templates": held_out},
    )
    return path


if __name__ == "__main__":
    main()
