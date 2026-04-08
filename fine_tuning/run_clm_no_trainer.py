import argparse
import json
import logging
import math
import os
import shutil
from itertools import chain
import datasets
from datasets import DatasetDict, load_dataset, load_from_disk
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import torch.distributed as dist

logger = get_logger(__name__)
require_version("datasets>=2.14.0", "To fix: pip install -r requirements.txt")


def delete_dir_if_exists(dir_path):
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            logger.info(f"Deleted directory: {dir_path}")
        except Exception as e:
            logger.warning(f"Failed to delete directory: {dir_path}, error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Finetune GPT-XL for causal language modeling (memory optimized version)")
    parser.add_argument("--train_file", type=str, required=True, help="Training data path (file or Arrow folder)")
    parser.add_argument("--validation_file", type=str, required=True, help="Validation data path (file or Arrow folder)")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite cache")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Pretrained model path")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Tokenizer path")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="Use slow Tokenizer")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Validation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Maximum training steps (overrides epochs)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                 "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=None, help="Warmup steps (10% of total steps by default)")
    parser.add_argument("--output_dir", type=str, required=True, help="Model save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpointing_steps", type=str, default="epoch", choices=["epoch", "step"],
                        help="Checkpoint save frequency")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--best_checkpoint_name", type=str, default="best_checkpoint", help="Best checkpoint name")
    parser.add_argument("--keep_final_model", action="store_true", default=False, help="Whether to keep final model")
    parser.add_argument("--block_size", type=int, default=128, help="Sequence length (memory optimized version)")
    parser.add_argument("--preprocessing_num_workers", type=int, default=2, help="Number of preprocessing processes")
    parser.add_argument("--no_keep_linebreaks", action="store_true", help="Do not keep line breaks (TXT files)")
    parser.add_argument("--logging_steps", type=int, default=50, help="Print training logs every N steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")

    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, ".gitignore"), "w") as f:
            f.write("step_*\nepoch_*\n")
            if not args.keep_final_model:
                f.write("final_model/\n")
            f.write(f"!{args.best_checkpoint_name}/\n")
    accelerator.wait_for_everyone()

    raw_datasets = DatasetDict()
    args.train_is_arrow = os.path.isdir(args.train_file) if args.train_file else False
    args.validation_is_arrow = os.path.isdir(args.validation_file) if args.validation_file else False

    if args.train_file:
        if args.train_is_arrow:
            raw_datasets["train"] = load_from_disk(args.train_file)
            logger.info(f"Loaded Arrow training set: {args.train_file}, total {len(raw_datasets['train'])} samples")
        else:
            ext = args.train_file.split(".")[-1]
            data_files = {"train": args.train_file}
            if ext == "txt":
                ext = "text"
                dataset_args = {"keep_linebreaks": not args.no_keep_linebreaks}
            else:
                dataset_args = {}
            loaded_ds = load_dataset(ext, data_files=data_files, **dataset_args)
            raw_datasets.update(loaded_ds)

    if args.validation_file:
        if args.validation_is_arrow:
            raw_datasets["validation"] = load_from_disk(args.validation_file)
            logger.info(f"Loaded Arrow validation set: {args.validation_file}, total {len(raw_datasets['validation'])} samples")
        else:
            ext = args.validation_file.split(".")[-1]
            data_files = {"validation": args.validation_file}
            if ext == "txt":
                ext = "text"
                dataset_args = {"keep_linebreaks": not args.no_keep_linebreaks}
            else:
                dataset_args = {}
            loaded_ds = load_dataset(ext, data_files=data_files, **dataset_args)
            raw_datasets["validation"] = loaded_ds["train"]

    logger.info(f"Dataset loading completed: training set {len(raw_datasets['train'])} samples, validation set {len(raw_datasets['validation'])} samples")

    tokenizer_name_or_path = args.tokenizer_name or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=not args.use_slow_tokenizer,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning(f"Automatically set pad_token to eos_token: {tokenizer.eos_token}")

    def tokenize_function(examples):
        text_key = "text" if "text" in examples else list(examples.keys())[0]
        if isinstance(examples[text_key], list) and isinstance(examples[text_key][0], str):
            concatenated = "\n".join(examples[text_key])
        else:
            concatenated = str(examples[text_key])
        return tokenizer(concatenated, truncation=False, add_special_tokens=False)

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_len = len(concatenated[list(examples.keys())[0]])
        total_len = (total_len // args.block_size) * args.block_size
        result = {
            k: [t[i:i + args.block_size] for i in range(0, total_len, args.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    if "input_ids" in raw_datasets["train"].column_names and \
            "attention_mask" in raw_datasets["train"].column_names and \
            "labels" in raw_datasets["train"].column_names:
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["validation"]
        logger.info("Using preprocessed dataset, skipping tokenization step")
    else:
        logger.info("Starting data preprocessing...")
        tokenized_ds = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers if (
                        args.preprocessing_num_workers and accelerator.is_main_process) else 0,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Tokenizing data"
        )
        processed_ds = tokenized_ds.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers if (
                        args.preprocessing_num_workers and accelerator.is_main_process) else 0,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts by block_size={args.block_size}"
        )
        train_dataset = processed_ds["train"]
        eval_dataset = processed_ds["validation"]
    logger.info(f"Data preparation completed: training set {len(train_dataset)} samples, validation set {len(eval_dataset)} samples")

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    model.to(accelerator.device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Manually enabled gradient checkpointing (memory optimization)")

    for name, param in model.named_parameters():
        if param.ndim == 1 and "bias" in name:
            param.requires_grad = False
    logger.info(f"Model loading completed: {args.model_name_or_path}")
    logger.info(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = None
    if accelerator.is_main_process:
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
        )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    args.num_warmup_steps = args.num_warmup_steps or int(args.max_train_steps * 0.1)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    best_perplexity = float("inf")
    global_step = 0
    resume_step = 0
    best_checkpoint_path = os.path.join(args.output_dir, args.best_checkpoint_name)
    training_logs = []
    log_file_path = os.path.join(args.output_dir, "training_log.json")

    if args.resume_from_checkpoint:
        if accelerator.is_main_process:
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        resume_step = accelerator.step
        global_step = resume_step
        if accelerator.is_main_process and os.path.exists(log_file_path):
            with open(log_file_path, "r") as f:
                logs = json.load(f)
                for log in logs:
                    if "perplexity" in log and log["perplexity"] < best_perplexity:
                        best_perplexity = log["perplexity"]
            logger.info(f"Resume completed: current step {resume_step}, best perplexity {best_perplexity:.2f}")

    logger.info("=" * 50)
    logger.info(f"Starting training ({args.num_train_epochs} epochs)")
    logger.info(
        f"Configuration: block_size={args.block_size} | batch_size={args.per_device_train_batch_size} | gradient accumulation={args.gradient_accumulation_steps}")
    logger.info("=" * 50)

    for epoch in range(args.num_train_epochs):
        model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{args.num_train_epochs}",
            disable=not accelerator.is_local_main_process,
        )
        for step, batch in enumerate(train_pbar):
            if global_step < resume_step:
                global_step += 1
                train_pbar.update(1)
                continue

            outputs = model(**batch)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if accelerator.is_main_process and global_step % args.logging_steps == 0:
                    avg_loss = total_train_loss / args.logging_steps
                    current_lr = lr_scheduler.get_last_lr()[0]
                    logger.info(f"Step {global_step} | Epoch {epoch + 1} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
                    training_logs.append({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": avg_loss,
                        "lr": current_lr
                    })
                    with open(log_file_path, "w") as f:
                        json.dump(training_logs, f, indent=2)
                    total_train_loss = 0.0

            train_pbar.update(1)
            if global_step >= args.max_train_steps:
                break
        train_pbar.close()

        if args.checkpointing_steps == "epoch" and accelerator.is_main_process:
            temp_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
            accelerator.save_state(temp_path)
            accelerator.wait_for_everyone()

            logger.info(f"Temporarily saved Epoch Checkpoint: {temp_path}")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.eval()
            total_eval_loss = 0.0

            with torch.no_grad():
                for eval_batch in tqdm(eval_dataloader, desc="Validating"):
                    eval_batch = {k: v.to(accelerator.device) for k, v in eval_batch.items()}
                    eval_outputs = unwrapped_model(**eval_batch)
                    total_eval_loss += eval_outputs.loss.item()

            avg_eval_loss = total_eval_loss / len(eval_dataloader)
            current_perplexity = math.exp(avg_eval_loss)
            logger.info(f"Epoch {epoch + 1} Validation | Loss: {avg_eval_loss:.4f} | Perplexity: {current_perplexity:.2f}")
            training_logs.append({
                "step": global_step,
                "epoch": epoch + 1,
                "eval_loss": avg_eval_loss,
                "perplexity": current_perplexity
            })
            with open(log_file_path, "w") as f:
                json.dump(training_logs, f, indent=2)

            if current_perplexity < best_perplexity:
                delete_dir_if_exists(best_checkpoint_path)
                shutil.copytree(temp_path, best_checkpoint_path)
                logger.info(f"Updated best checkpoint: {best_checkpoint_path} (perplexity {current_perplexity:.2f})")
                best_perplexity = current_perplexity

            delete_dir_if_exists(temp_path)
            unwrapped_model.train()

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    logger.info("=" * 50)
    logger.info("Training completed!")

    if args.keep_final_model and accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "final_model")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            final_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(final_path)
        logger.info(f"Final model saved to: {final_path}")

    if accelerator.is_main_process:
        for item in os.listdir(args.output_dir):
            item_path = os.path.join(args.output_dir, item)
            if os.path.isdir(item_path) and (item.startswith("step_") or item.startswith("epoch_")):
                delete_dir_if_exists(item_path)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(f"Training summary: total steps {global_step} | best perplexity {best_perplexity:.2f}")
        logger.info(f"Best checkpoint path: {best_checkpoint_path}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()