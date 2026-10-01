"""Phase 4: LoRA/QLoRA fine-tuning via Unsloth (with transformers+PEFT fallback).

Trains the model to output both the Triton kernel and the optimization
explanation in a structured format. Run on Kaggle/Colab with a T4 GPU.

    python training/finetune.py --config training/configs/finetune_default.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kernelforge.dataset import load_training_data, make_split, template_of
from kernelforge.prompts import SYSTEM_PROMPT, format_training_example

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"


def _format_chat_example(entry: dict, tokenizer) -> dict:
    """Build a tokenized chat example for SFT."""
    example = format_training_example(entry)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text, "id": entry["id"], "category": entry["category"]}


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def finetune_unsloth(config: dict, train_entries: list[dict], val_entries: list[dict], output_dir: Path) -> None:
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    train_ds = Dataset.from_list([_format_chat_example(e, tokenizer) for e in train_entries])
    val_ds = Dataset.from_list([_format_chat_example(e, tokenizer) for e in val_entries]) if val_entries else None

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        num_train_epochs=config.get("num_epochs", 3),
        learning_rate=config.get("learning_rate", 2e-4),
        weight_decay=config.get("weight_decay", 0.01),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds else "no",
        fp16=True,
        seed=config.get("seed", 42),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        args=training_args,
    )

    trainer.train()
    model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))


def finetune_transformers(config: dict, train_entries: list[dict], val_entries: list[dict], output_dir: Path) -> None:
    """Fallback when Unsloth is unavailable: QLoRA on a CUDA GPU (Kaggle/Colab, where
    Unsloth fights the preinstalled torch), or fp32 LoRA on CPU for a dev-machine smoke run."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

    cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"], torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True,
        )
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=config["max_seq_length"])

    train_ds = Dataset.from_list([_format_chat_example(e, tokenizer) for e in train_entries]).map(tokenize)
    val_ds = (
        Dataset.from_list([_format_chat_example(e, tokenizer) for e in val_entries]).map(tokenize)
        if val_entries
        else None
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        num_train_epochs=config.get("num_epochs", 3) if cuda else 1,
        learning_rate=config.get("learning_rate", 2e-4),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        weight_decay=config.get("weight_decay", 0.01),
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch" if val_ds is not None else "no",
        fp16=cuda,
        gradient_checkpointing=cuda,
        optim="paged_adamw_8bit" if cuda else "adamw_torch",
        seed=config.get("seed", 42),
        report_to="none",
        use_cpu=not cuda,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a code model on verified Op->Kernel+Explanation pairs.")
    parser.add_argument("--config", type=Path, default=Path("training/configs/finetune_default.json"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    entries = load_training_data(args.dataset)
    if not entries:
        print("No training data found.", file=sys.stderr)
        sys.exit(1)

    strategy = config.get("split", "template")
    ratios = {"val_ratio": config.get("val_ratio", 0.10)}
    if strategy == "random":
        ratios.update(train_ratio=config.get("train_ratio", 0.85), test_ratio=config.get("test_ratio", 0.05))
    train, val, test = make_split(entries, strategy, seed=config.get("seed", 42), **ratios)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (CHECKPOINTS_DIR / f"run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_info = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "split": strategy,
        "held_out_templates": sorted({template_of(e) for e in test}),
        "config": config,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))

    # Save held-out test set for Phase 5 eval
    with open(output_dir / "held_out_test.jsonl", "w") as f:
        for entry in test:
            f.write(json.dumps(entry) + "\n")

    print(f"Training on {len(train)} examples, validating on {len(val)}, held-out test: {len(test)}")
    print(f"Checkpoints -> {output_dir}")

    try:
        finetune_unsloth(config, train, val, output_dir)
        print("Training complete (Unsloth backend).")
    except ImportError:
        print("Unsloth not available — using transformers+PEFT CPU fallback (dev only).")
        finetune_transformers(config, train, val, output_dir)
        print("Training complete (transformers fallback).")


if __name__ == "__main__":
    main()
