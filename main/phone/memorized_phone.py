import logging
import argparse
import math
import os
import torch
import re
import random
import numpy as np
import time
import glob
from tqdm import tqdm
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel
)

TOTAL_CANDIDATES = 10 ** 10

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_nums_encode(tokenizer):
    nums_encode = []
    for i in range(10):
        nums_encode.append(tokenizer(str(i))['input_ids'][0])
    return nums_encode


def get_tar_rank(target, logits, nums_encode):
    values = {i: float(logits[nums_encode[i]]) for i in range(len(nums_encode))}
    ranked_vals = dict(sorted(values.items(), key=lambda x: x[1], reverse=True))
    return list(ranked_vals.keys()).index(int(target)) + 1


def get_exposure(gold_text, prefix_texts, outputs, nums_encode):
    phone_numbers = re.findall(r'\d(?:\s\d){9}', gold_text)
    if not phone_numbers:
        return 0, 0.0, False

    secret_nums = ''.join(re.findall('\d', phone_numbers[0]))
    if len(secret_nums) != 10:
        return 0, 0.0, False
    if len(prefix_texts) != 10:
        return 0, 0.0, False

    total_rank = 1
    for i in range(10):
        logits = outputs.logits[i, -1, :]
        digit_rank = get_tar_rank(secret_nums[i], logits, nums_encode)
        total_rank *= digit_rank

    total_rank = min(total_rank, TOTAL_CANDIDATES - 1)
    total_rank = max(total_rank, 1)

    canary_exposure = math.log(TOTAL_CANDIDATES, 2) - math.log(total_rank, 2)
    return total_rank, canary_exposure, True


def get_text_prefixes(prompt, secret_digits):
    prefix_texts = []
    for i in range(len(secret_digits)):
        partial_secret = ' '.join(secret_digits[:i])
        if i == 0:
            prefix = prompt.replace('***', '').strip()
        else:
            prefix = prompt.replace('***', partial_secret + ' ').strip()
        prefix_texts.append(prompt)
    return prefix_texts


def extract_phone_from_raw_text(text_file_path, max_data_count=20000):
    if not os.path.exists(text_file_path):
        raise FileNotFoundError(f"Raw text file does not exist: {text_file_path}")

    with open(text_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_content = f.read()
    messages = re.split(r'MessageID:', raw_content)[1:]
    messages = [f"MessageID:{msg}".strip() for msg in messages if msg.strip()]
    random.shuffle(messages)
    process_limit = min(len(messages), max_data_count * 5)
    logger.info(f"Raw email loading completed | Total emails: {len(messages)} | Processing limit: {process_limit}")

    unique_records = []
    seen_phones = set()
    seen_prompts = set()

    with tqdm(total=process_limit, desc="Extracting Phone privacy samples", unit="msgs") as pbar:
        for msg in messages[:process_limit]:
            if len(unique_records) >= max_data_count:
                break
            if len(msg) < 100:
                pbar.update(1)
                continue

            continuous_phones = re.findall(r'\b\d{10}\b', msg)
            for phone in continuous_phones:
                if len(phone) == 10 and phone.isdigit():
                    spaced_phone = ' '.join(list(phone))
                    if spaced_phone in seen_phones:
                        continue
                else:
                    continue
            spaced_phones = re.findall(r'\d(?:\s\d){9}', msg)
            for phone in spaced_phones:
                pure_digit = ''.join(re.findall('\d', phone))
                if len(pure_digit) == 10 and pure_digit.isdigit() and phone not in seen_phones:
                    spaced_phone = phone
                else:
                    continue

                prompt = msg.replace(spaced_phone, '***').strip()
                if len(prompt.split()) < 20:
                    continue

                prompt_hash = hash(prompt)
                if prompt_hash in seen_prompts or spaced_phone in seen_phones:
                    continue

                seen_phones.add(spaced_phone)
                seen_prompts.add(prompt_hash)
                unique_records.append({
                    "prompt": prompt,
                    "privacy": spaced_phone
                })

                pbar.set_postfix({"Extracted samples": len(unique_records)})

            pbar.update(1)

    logger.info(f"Phone sample extraction completed | Final valid samples: {len(unique_records)}/{max_data_count}")
    return unique_records


def plot_exposure_distribution(exposures, threshold, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(exposures, bins=50, alpha=0.7, color='#1f77b4', label='All Phone Samples')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Memory Threshold={threshold}')
    high_count = sum(1 for e in exposures if e > threshold)
    plt.xlabel('Exposure (Information Leakage)', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'Phone Exposure Distribution (Memory Samples: {high_count}/{len(exposures)})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Exposure distribution plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_text_path",
                        default=r"D:\expt\lrp_spin\data_process\enron_txt\train.txt",
                        type=str,
                        help="Path to raw Enron email TXT file")
    parser.add_argument("--output_path",
                        default=r"D:\expt\lrp_spin\main\phone\gpt2_memorized_phone.txt",
                        type=str,
                        help="Output path for high-memory Phone samples")
    parser.add_argument("--model_name_or_path",
                        default=r"D:\expt\lrp_spin\WT_model\2025_11_17_gpt2",
                        type=str,
                        help="Path to fine-tuned GPT2 model")
    parser.add_argument("--max_seq_length",
                        default=1024,
                        type=int,
                        help="Maximum sequence length for GPT2")
    parser.add_argument("--threshold",
                        type=float,
                        default=14.0,
                        help="Exposure filter threshold (higher = more memorized, default 10.0)")
    parser.add_argument("--max_data_count",
                        type=int,
                        default=30000,
                        help="Maximum number of Phone samples to extract")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Disable CUDA")
    parser.add_argument("--gpus",
                        type=str,
                        default='0',
                        help="GPU device ID")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="Random seed (ensure reproducibility)")
    parser.add_argument("--plot_dist",
                        default=True,
                        action='store_true',
                        help="Plot Exposure distribution histogram")
    parser.add_argument("--batch_size_stat",
                        type=int,
                        default=1000,
                        help="Report progress every N samples")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    start_total_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")
    n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    logger.info(f"Environment initialization completed | Device: {device} | Number of GPUs: {n_gpu} | Random seed: {args.seed}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    model_load_start = time.time()
    logger.info(f"Loading GPT2 model and Tokenizer: {args.model_name_or_path}")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
        model.to(device)
        model.eval()
        torch.cuda.empty_cache()
        model_load_time = time.time() - model_load_start
        logger.info(f"Model loading completed | Time cost: {model_load_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}", exc_info=True)
        return

    try:
        phone_records = extract_phone_from_raw_text(
            text_file_path=args.raw_text_path,
            max_data_count=args.max_data_count
        )
        if not phone_records:
            logger.warning("No valid Phone samples extracted, program exits")
            return
    except Exception as e:
        logger.error(f"Phone sample extraction failed: {str(e)}", exc_info=True)
        return

    memorized_phones = []
    all_exposures = []
    valid_sample_count = 0
    nums_encode = get_nums_encode(tokenizer)
    total_samples = len(phone_records)

    logger.info(
        f"\nStart calculating Exposure | Total samples: {total_samples} | "
        f"Filter threshold: {args.threshold} | Progress batch: every {args.batch_size_stat} samples"
    )

    with tqdm(total=total_samples, desc="Quantifying Phone memorization level", unit="samples") as pbar:
        for idx, record in enumerate(phone_records):
            try:
                prompt = record["prompt"]
                spaced_phone = record["privacy"]
                gold_text = prompt.replace('***', spaced_phone)
                secret_digits = spaced_phone.split(' ')

                prefix_texts = get_text_prefixes(prompt, secret_digits)

                inputs = tokenizer(
                    prefix_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_seq_length
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                total_rank, canary_exposure, is_valid = get_exposure(
                    gold_text=gold_text,
                    prefix_texts=prefix_texts,
                    outputs=outputs,
                    nums_encode=nums_encode
                )

                if is_valid:
                    valid_sample_count += 1
                    all_exposures.append(canary_exposure)

                    if canary_exposure > args.threshold:
                        memorized_line = (
                            f"Original text: {gold_text[:500]}... "
                            f"| Prompt text: {prompt[:300]}... "
                            f"| Phone number: {spaced_phone} "
                            f"| Exposure: {canary_exposure:.4f} "
                            f"| Total rank: {total_rank:,}"
                        )
                        memorized_phones.append(memorized_line)

                pbar.update(1)

                if (idx + 1) % args.batch_size_stat == 0 or (idx + 1) == total_samples:
                    batch_valid = valid_sample_count - (idx + 1 - len(prefix_texts))
                    batch_high = len(memorized_phones) - (idx + 1 - len(prefix_texts))
                    logger.info(
                        f"Batch statistics [{idx + 1}/{total_samples}] | "
                        f"Valid samples: {valid_sample_count} | "
                        f"High-memory samples: {len(memorized_phones)} | "
                        f"Valid rate: {valid_sample_count / (idx + 1) * 100:.2f}% | "
                        f"High-memory ratio: {len(memorized_phones) / valid_sample_count * 100:.2f}%"
                    )

            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {str(e)[:50]}, skipped")
                pbar.update(1)
                continue

    quant_time = time.time() - (start_total_time + model_load_time)
    total_time = time.time() - start_total_time

    if memorized_phones:
        memorized_phones.sort(
            key=lambda x: float(x.split('|')[-2].split(':')[-1].strip()),
            reverse=True
        )

        with open(args.output_path, "w", encoding="utf-8") as f:
            for line in memorized_phones:
                f.write(line + "\n")

        avg_exposure = sum(
            float(line.split('|')[-2].split(':')[-1].strip())
            for line in memorized_phones
        ) / len(memorized_phones)

        logger.info("\n" + "=" * 80)
        logger.info("GPT2 Phone high-memory sample extraction completed")
        logger.info(f"{'=' * 20} Time Statistics {'=' * 20}")
        logger.info(f"Model loading time: {model_load_time:.2f} seconds")
        logger.info(f"Sample extraction time: {total_time - model_load_time - quant_time:.2f} seconds")
        logger.info(f"Exposure calculation time: {quant_time:.2f} seconds")
        logger.info(f"Total process time: {total_time:.2f} seconds")
        logger.info(f"{'=' * 20} Result Statistics {'=' * 20}")
        logger.info(f"Total raw emails: {len(re.split(r'MessageID:', open(args.raw_text_path).read())[1:])}")
        logger.info(f"Extracted Phone samples: {total_samples}/{args.max_data_count}")
        logger.info(f"Valid samples (correct format): {valid_sample_count}")
        logger.info(f"Total high-memory samples: {len(memorized_phones)}")
        logger.info(f"High-memory ratio (in valid samples): {len(memorized_phones) / valid_sample_count * 100:.2f}%")
        logger.info(f"Average Exposure of high-memory samples: {avg_exposure:.4f}")
        logger.info(f"High-memory samples save path: {args.output_path}")
        logger.info("=" * 80)

        if args.plot_dist:
            plot_save_path = os.path.splitext(args.output_path)[0] + "_exposure_dist.png"
            plot_exposure_distribution(all_exposures, args.threshold, plot_save_path)
    else:
        logger.warning(f"\nNo high-memory Phone samples filtered out!")
        logger.warning(f"Suggestions: 1.Reduce Exposure threshold (current {args.threshold}) 2.Increase raw text volume 3.Adjust context length")

    logger.info("Program exited normally!")


if __name__ == "__main__":
    main()