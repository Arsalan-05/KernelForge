"""Phase 4: LoRA/QLoRA fine-tuning via Unsloth (with transformers+PEFT fallback).

Trains the model to output both the Triton kernel and the optimization
explanation in a structured format. Run on Kaggle/Colab with a T4 GPU.

    python training/finetune.py --config training/configs/finetune_default.json

CPU plumbing check (tiny random-weight model, 2 optimizer steps, no GPU):

    python training/finetune.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kernelforge.dataset import load_eval_entries, make_split, template_of
from kernelforge.model import encode_text, has_chat_template, render_prompt, unsloth_available
from kernelforge.prompts import build_messages, format_response

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
IGNORE_INDEX = -100  # label value the cross-entropy loss skips


def build_sft_features(entry: dict, tokenizer, max_seq_length: int) -> dict | None:
    """Tokenize one entry so that only the assistant's answer is trained on.

    The prompt is rendered by the same `render_prompt` the inference code
    uses, so train and test prompts are identical. Prompt tokens get label
    -100 (no loss); the answer tokens, including the end-of-turn token that
    teaches the model to stop, keep their ids as labels.

    Returns None when the example is longer than `max_seq_length`: truncating
    would cut the kernel or the explanation off mid-way, and training on that
    teaches the model to stop early.
    """
    if not has_chat_template(tokenizer):
        raise ValueError("fine-tuning needs a tokenizer with a chat template (use an -Instruct model)")
    messages = build_messages(entry["description"], entry["pytorch_reference"])
    prompt_text = render_prompt(tokenizer, messages)
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": format_response(entry)}], tokenize=False
    )
    if not full_text.startswith(prompt_text):
        raise ValueError(f"{entry['id']}: chat template does not extend the generation prompt; "
                         "cannot locate the assistant turn for loss masking")

    # Tokenize prompt and answer separately, exactly as inference sees them:
    # the prompt is encoded alone and the model generates everything after it.
    prompt_ids = encode_text(tokenizer, prompt_text)
    answer_ids = tokenizer(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_seq_length:
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids,
    }


def build_sft_dataset(entries: list[dict], tokenizer, max_seq_length: int):
    """Returns (features, ids of examples dropped for being too long)."""
    features, dropped = [], []
    for entry in entries:
        feat = build_sft_features(entry, tokenizer, max_seq_length)
        if feat is None:
            dropped.append(entry["id"])
        else:
            features.append(feat)
    return features, dropped


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def load_unsloth(config: dict):
    from unsloth import FastLanguageModel  # import before transformers so its patches apply

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config.get("load_in_4bit", True),
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=LORA_TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def load_transformers(config: dict):
    """Load model + LoRA adapter for training.

    CUDA:  QLoRA (4-bit NF4, fp16 activations) — Kaggle/Colab T4 path.
    MPS:   fp32 LoRA (no quantization; bitsandbytes not available on MPS) — Mac pilot path.
    CPU:   fp32 LoRA — dry-run / CI path.
    """
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    mps  = (not cuda) and torch.backends.mps.is_available()
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)

    if cuda:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ) if config.get("load_in_4bit", True) else None
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"], quantization_config=quant, torch_dtype=torch.float16,
            device_map={"": 0}, trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif mps:
        # MPS: load in fp32 to CPU first, then move to MPS.
        # device_map="mps" is not reliably supported by all PEFT versions;
        # loading to CPU then .to("mps") is the stable approach.
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"], torch_dtype=torch.float32,
            device_map=None, trust_remote_code=True,
        )
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"], torch_dtype=torch.float32,
            device_map="cpu", trust_remote_code=True,
        )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def make_gpu_log_callback(log_path: Path):
    """TrainerCallback: appends loss, tok/s, and GPU memory to a JSONL file each step."""
    import torch
    from transformers import TrainerCallback

    class GpuLogCallback(TrainerCallback):
        _last_t: float = 0.0
        _last_step: int = 0

        def on_step_end(self, args, state, control, **kwargs):
            now = time.monotonic()
            step = state.global_step
            loss = None
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    loss = entry["loss"]
                    break
            tps: float | None = None
            if self._last_t and step > self._last_step:
                elapsed = now - self._last_t
                token_estimate = args.per_device_train_batch_size * args.gradient_accumulation_steps * 512
                tps = token_estimate / elapsed if elapsed > 0 else None
            self._last_t = now
            self._last_step = step
            mem_gb: float | None = None
            if torch.cuda.is_available():
                mem_gb = round(torch.cuda.memory_allocated() / 1024 ** 3, 3)
            record = {"step": step, "loss": loss, "tokens_per_second": tps, "gpu_memory_allocated_gb": mem_gb}
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            return control

    return GpuLogCallback()


def make_time_limit_callback(deadline: float | None, adapter_dir: Path):
    """TrainerCallback for --max-minutes: at the first optimizer step past the
    deadline (time.monotonic()), save the adapter and stop training cleanly."""
    from transformers import TrainerCallback

    class TimeLimitCallback(TrainerCallback):
        stopped_at_step: int | None = None

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if deadline is not None and self.stopped_at_step is None and time.monotonic() >= deadline:
                model.save_pretrained(str(adapter_dir))
                self.stopped_at_step = state.global_step
                control.should_training_stop = True
                print(f"--max-minutes reached at step {state.global_step}/{state.max_steps}; "
                      f"adapter saved to {adapter_dir}")
            return control

    return TimeLimitCallback()


def train(config: dict, backend: str, train_entries: list[dict], val_entries: list[dict],
          output_dir: Path, max_steps: int = -1, deadline: float | None = None,
          resume_from_checkpoint: str | None = None) -> dict:
    """Load the model, train with assistant-only loss, save the adapter. Returns run stats."""
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    model, tokenizer = load_unsloth(config) if backend == "unsloth" else load_transformers(config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # padding only; labels come from build_sft_features
    model.print_trainable_parameters()

    max_len = config["max_seq_length"]
    train_feats, dropped_train = build_sft_dataset(train_entries, tokenizer, max_len)
    val_feats, dropped_val = build_sft_dataset(val_entries, tokenizer, max_len)
    if dropped_train or dropped_val:
        print(f"WARNING: dropped {len(dropped_train)} train / {len(dropped_val)} val examples longer "
              f"than max_seq_length={max_len}: {dropped_train + dropped_val}")
    if not train_feats:
        raise SystemExit("No training examples left after tokenization.")
    supervised = sum(sum(1 for t in f["labels"] if t != IGNORE_INDEX) for f in train_feats)
    total = sum(len(f["input_ids"]) for f in train_feats)
    longest = max(len(f["input_ids"]) for f in train_feats)
    print(f"Supervised (answer) tokens: {supervised}/{total} ({supervised / total:.0%}); "
          f"longest example: {longest} tokens")

    cuda = torch.cuda.is_available()
    mps  = (not cuda) and torch.backends.mps.is_available()
    dry  = max_steps > 0
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        num_train_epochs=config.get("num_epochs", 3),
        max_steps=max_steps,
        learning_rate=config.get("learning_rate", 2e-4),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        weight_decay=config.get("weight_decay", 0.01),
        logging_steps=1 if dry else 5,
        # Save every 50 optimizer steps so a crash loses at most 50 steps (~8 min on T4).
        # save_total_limit=3 keeps the 3 most recent checkpoints + the final adapter.
        save_strategy="no" if dry else "steps",
        save_steps=50,
        save_total_limit=3,
        eval_strategy=("steps" if dry else "steps") if val_feats else "no",
        eval_steps=1 if dry else 50,
        fp16=cuda,          # T4 supports fp16; MPS/CPU use fp32
        bf16=False,         # T4 has no bf16; Ampere+ only
        # Unsloth applies its own gradient checkpointing in get_peft_model.
        gradient_checkpointing=cuda and backend == "transformers",
        optim="paged_adamw_8bit" if cuda and backend == "transformers" else "adamw_torch",
        seed=config.get("seed", 42),
        report_to="none",
        # use_cpu=True forces CPU; on MPS we want MPS (not CPU, not CUDA).
        use_cpu=not (cuda or mps),
    )
    adapter_dir = output_dir / "adapter"
    time_limit = make_time_limit_callback(deadline, adapter_dir)
    gpu_log = make_gpu_log_callback(output_dir / "gpu_training_log.jsonl")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_feats),
        eval_dataset=Dataset.from_list(val_feats) if val_feats else None,
        # Pads input_ids with pad_token_id and labels with -100 (only matters for batch > 1).
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX),
        callbacks=[time_limit, gpu_log],
    )
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finally:
        # Loss history goes in the run folder itself (not only in checkpoint-*),
        # and is written even if training crashes (e.g. out of memory).
        (output_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
        trainer.save_state()  # -> <run>/trainer_state.json

    # Record peak GPU memory in GB; skipped safely when no GPU is present.
    peak_memory_gb: float | None = None
    if cuda and torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return {
        "partial": time_limit.stopped_at_step is not None,
        "stopped_at_step": time_limit.stopped_at_step,
        "steps_planned": trainer.state.max_steps,
        "backend": backend,
        "dropped_too_long": dropped_train + dropped_val,
        "supervised_token_fraction": supervised / total,
        "longest_example_tokens": longest,
        "peak_memory_gb": peak_memory_gb,
    }


def build_tiny_random_model(tokenizer_name: str, path: Path) -> Path:
    """A 2-layer random-weight Qwen2 using the configured model's tokenizer.

    For --dry-run only: it exercises tokenization, masking, LoRA, the Trainer
    and adapter save/load on CPU in seconds. Its outputs are noise.
    """
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    if (path / "config.json").exists():
        return path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    config = Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
        tie_word_embeddings=True, eos_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(0)
    Qwen2ForCausalLM(config).save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a code model on verified Op->Kernel+Explanation pairs.")
    parser.add_argument("--config", type=Path, default=Path("training/configs/finetune_default.json"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=["auto", "unsloth", "transformers"], default="auto",
                        help="auto = Unsloth if it imports cleanly, else transformers + PEFT")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Stop training at the first optimizer step after this long (counted from "
                             "script start), save the adapter, and mark the run partial")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, metavar="PATH",
                        help="Path to a Trainer checkpoint directory (checkpoint-NNN) to resume from. "
                             "Pass 'last' to resume from the most recent checkpoint in --output-dir.")
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU plumbing check: tiny random model, 8 train examples, 2 steps")
    args = parser.parse_args()

    deadline = None if args.max_minutes is None else time.monotonic() + args.max_minutes * 60
    resume_ckpt = args.resume_from_checkpoint
    config = _load_config(args.config)
    entries, source = load_eval_entries(args.dataset, dry_run=args.dry_run)
    if not entries:
        print("No verified training data found (data/verified/dataset.jsonl is missing or empty). "
              "Run data/build_dataset.py --device cuda first, or use --dry-run.", file=sys.stderr)
        sys.exit(1)

    strategy = config.get("split", "template")
    ratios = {"val_ratio": config.get("val_ratio", 0.10)}
    if strategy == "random":
        ratios.update(train_ratio=config.get("train_ratio", 0.85), test_ratio=config.get("test_ratio", 0.05))
    train_set, val_set, test_set = make_split(entries, strategy, seed=config.get("seed", 42), **ratios)
    if not train_set:
        print("The split left no training examples (each category needs at least 2 templates).", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (CHECKPOINTS_DIR / f"{'dryrun' if args.dry_run else 'run'}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_info = {
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set),
        "split": strategy,
        "held_out_templates": sorted({template_of(e) for e in test_set}),
        "data_source": source,
        "dry_run": args.dry_run,
        "config": config,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    # Written before training so a crashed run still records its split.
    (output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))
    # Save held-out test set for Phase 5 eval
    with open(output_dir / "held_out_test.jsonl", "w") as f:
        for entry in test_set:
            f.write(json.dumps(entry) + "\n")

    max_steps = -1
    backend = args.backend
    if args.dry_run:
        config = {**config, "model_name": str(build_tiny_random_model(config["model_name"], output_dir / "tiny_random_model"))}
        train_set, val_set, max_steps, backend = train_set[:8], val_set[:2], 2, "transformers"
        # Config for running evaluation/finetuned_eval.py against this dry-run adapter on CPU.
        eval_cfg = {"model_name": config["model_name"], "max_new_tokens": 16, "load_in_4bit": False,
                    "max_seq_length": config["max_seq_length"]}
        (output_dir / "dry_run_eval_config.json").write_text(json.dumps(eval_cfg, indent=2))
    elif backend == "auto":
        backend = "unsloth" if unsloth_available() else "transformers"

    print(f"Data: {source}. Training on {len(train_set)} examples, validating on {len(val_set)}, "
          f"held-out test: {len(test_set)}; backend: {backend}")
    print(f"Checkpoints -> {output_dir}")
    # Handle "last" shorthand: find the most recent checkpoint-NNN in output_dir.
    if resume_ckpt == "last":
        ckpts = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        resume_ckpt = str(ckpts[-1]) if ckpts else None
        if resume_ckpt:
            print(f"Resuming from {resume_ckpt}")
        else:
            print("--resume-from-checkpoint last: no checkpoint found in output_dir; starting fresh.")
    split_info.update(train(config, backend, train_set, val_set, output_dir, max_steps=max_steps,
                            deadline=deadline, resume_from_checkpoint=resume_ckpt))
    (output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))
    status = (f"stopped early at step {split_info['stopped_at_step']}/{split_info['steps_planned']} (PARTIAL)"
              if split_info["partial"] else "complete")
    print(f"Training {status} ({backend} backend). Adapter -> {output_dir / 'adapter'}")


if __name__ == "__main__":
    main()
