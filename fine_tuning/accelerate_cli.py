import sys
from argparse import ArgumentParser
from accelerate.commands.config import config_command_parser
from accelerate.commands.env import env_command_parser
from accelerate.commands.launch import launch_command_parser
from accelerate.commands.test import test_command_parser
from accelerate.commands.tpu import tpu_command_parser


def main():
    default_args = [
        "launch",
        "--num_machines", "1",
        "--num_processes", "4",
        "--main_process_port", "12357",
        "--mixed_precision", "bf16",
        "/root/autodl-tmp/LRP_SPIN/fine_tuning/run_clm_no_trainer.py",
        "--train_file", "/root/autodl-tmp/LRP_SPIN/data/train/final",
        "--validation_file", "/root/autodl-tmp/LRP_SPIN/data/valid/final",
        "--overwrite_cache",
        "--model_name_or_path", "/root/autodl-tmp/LRP_SPIN/base_model/GPT_XL",
        "--block_size", "512",
        "--gradient_checkpointing",
        "--per_device_train_batch_size", "12",
        "--per_device_eval_batch_size", "12",
        "--learning_rate", "5e-5",
        "--weight_decay", "0.01",
        "--num_train_epochs", "1",
        "--gradient_accumulation_steps", "8",
        "--lr_scheduler_type", "cosine",
        "--num_warmup_steps", "500",
        "--output_dir", "/root/autodl-tmp/LRP_SPIN/WT_model/2025_11_23_gpt2_xl",
        "--checkpointing_steps", "epoch",
        "--seed", "42",
        "--preprocessing_num_workers", "2",
        "--logging_steps", "50",
        "--best_checkpoint_name", "best_checkpoint",
        "--keep_final_model",
    ]

    if len(sys.argv) == 1:
        sys.argv.extend(default_args)

    parser = ArgumentParser("Accelerate CLI tool", usage="accelerate <command> [<args>]")
    subparsers = parser.add_subparsers(help="accelerate command helpers")
    config_command_parser(subparsers=subparsers)
    env_command_parser(subparsers=subparsers)
    launch_command_parser(subparsers=subparsers)
    tpu_command_parser(subparsers=subparsers)
    test_command_parser(subparsers=subparsers)
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)
    args.func(args)


if __name__ == "__main__":
    main()