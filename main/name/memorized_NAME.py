import logging
import argparse
import os
import time
import re
import torch
import random
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk import ne_chunk
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoConfig
)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
warning_counter = {}


def download_nltk_deps():
    required_deps = [
        ('tokenizers/punkt', 'punkt'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
        ('corpora/words', 'words')
    ]
    for path, dep in required_deps:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Download NLTK dependency: {dep}")
            nltk.download(dep)


def extract_english_names(text):
    names = []
    try:
        tokens = word_tokenize(text[:5000])
        tagged = pos_tag(tokens)
        entities = ne_chunk(tagged)

        current_name = []
        for chunk in entities:
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                name_parts = [c[0] for c in chunk]
                full_name = ' '.join(name_parts).strip()
                if (len(full_name) >= 3 and
                        any(c.isupper() for c in full_name) and
                        not any(c.isdigit() for c in full_name)):
                    if full_name not in names:
                        names.append(full_name)
    except Exception as e:
        logger.warning(f"Error extracting English names: {str(e)[:50]}")
    return names


def extract_phone_numbers(text):
    phones = []
    pattern = r'\b(\d\s+){9,}\d\b'
    matches = re.findall(pattern, text)
    for match in matches:
        phone = ''.join([c for c in match if c.isdigit()])
        if 10 <= len(phone) <= 15 and phone not in phones:
            phones.append(phone)
    return phones


def load_raw_text_data(text_file_path, max_data_count=10000):
    if not os.path.exists(text_file_path):
        raise FileNotFoundError(f"Raw text file does not exist: {text_file_path}")
    with open(text_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_content = f.read()
    messages = re.split(r'MessageID:', raw_content)[1:]
    messages = [f"MessageID:{msg}".strip() for msg in messages if msg.strip()]
    random.shuffle(messages)
    process_limit = min(len(messages), max_data_count * 5)
    logger.info(
        f"Raw text loading completed | Total messages: {len(messages)} | "
        f"Processing limit: {process_limit} (max extract {max_data_count} privacy records)"
    )
    return messages[:process_limit]


def get_privacy_from_raw_text(text_file_path, privacy_type, min_context_len=10, max_context_len=20,
                              max_data_count=10000):
    messages = load_raw_text_data(text_file_path, max_data_count)
    privacy_records = []
    seen_privacies = set()
    seen_contexts = set()
    total_processed = len(messages)
    start_time = time.time()

    with tqdm(total=total_processed, desc=f"Extracting {privacy_type} privacy", unit="msgs", ncols=100) as pbar:
        for idx, msg in enumerate(messages):
            if len(privacy_records) >= max_data_count:
                pbar.update(total_processed - idx)
                break
            if not msg or len(msg) < 100:
                pbar.update(1)
                continue

            try:
                if privacy_type == 'name':
                    privacies = extract_english_names(msg)
                elif privacy_type == 'phone':
                    privacies = extract_phone_numbers(msg)
                else:
                    raise ValueError(f"Unsupported privacy type: {privacy_type}")

                for privacy in privacies:
                    if privacy in seen_privacies:
                        continue
                    msg_parts = msg.split(privacy)
                    if len(msg_parts) < 2:
                        continue

                    prefix_words = msg_parts[0].strip().split()
                    suffix_words = msg_parts[1].strip().split()
                    if len(prefix_words) < min_context_len or len(suffix_words) < min_context_len:
                        continue
                    prefix_len = random.randint(min_context_len, min(max_context_len, len(prefix_words)))
                    suffix_len = random.randint(min_context_len, min(max_context_len, len(suffix_words)))
                    prefix = ' '.join(prefix_words[-prefix_len:]).strip()
                    suffix = ' '.join(suffix_words[:suffix_len]).strip()

                    context = f"{prefix} the target is {suffix}".strip()

                    if len(context.split()) < 20:
                        continue
                    context_hash = hash(context)
                    if context_hash in seen_contexts:
                        continue

                    seen_privacies.add(privacy)
                    seen_contexts.add(context_hash)
                    privacy_records.append({
                        "context": context,
                        "privacy": privacy,
                        "privacy_type": privacy_type
                    })

                    pbar.set_postfix({
                        f"Extracted {privacy_type}": len(privacy_records),
                        "Progress": f"{len(privacy_records)}/{max_data_count}"
                    })

                    if len(privacy_records) >= max_data_count:
                        pbar.update(total_processed - idx)
                        logger.info(f"Extracted {max_data_count} {privacy_type} privacy records, reached upper limit")
                        break

            except Exception as e:
                warning_key = "Text extraction error"
                warning_counter[warning_key] = warning_counter.get(warning_key, 0) + 1
                if warning_counter[warning_key] == 1:
                    logger.warning(f"Error processing message {idx}: {str(e)[:50]} (First occurrence, only count later)")
            finally:
                pbar.update(1)

    extract_time = time.time() - start_time
    privacy_to_record = {}
    for record in privacy_records:
        privacy = record["privacy"]
        if privacy not in privacy_to_record:
            privacy_to_record[privacy] = record
    unique_records = list(privacy_to_record.values())[:max_data_count]

    logger.info(
        f"{privacy_type} privacy extraction completed | Time consumed: {extract_time:.2f}s | "
        f"Processed messages: {min(idx + 1, total_processed)} | "
        f"Final extracted {privacy_type} count: {len(unique_records)}/{max_data_count}"
    )
    return unique_records, extract_time, len(messages)


def calculate_privacy_mrr(context, target_privacy, tokenizer, model, device, max_seq_length, privacy_type):
    target_tokens = tokenizer.tokenize(target_privacy)
    if not target_tokens:
        return 0.0

    context_ids = tokenizer.encode(
        context,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length - 10
    )
    context_ids = torch.tensor([context_ids]).to(device)

    total_mrr = 0.0
    valid_token_count = 0
    current_input_ids = context_ids.clone()

    for target_token in target_tokens:
        if current_input_ids.size(1) >= max_seq_length - 1:
            break

        with torch.no_grad():
            outputs = model(current_input_ids)
            logits = outputs.logits[:, -1, :]
            logits = logits / 0.7

        target_token_id = tokenizer.convert_tokens_to_ids(target_token)
        if target_token_id == tokenizer.unk_token_id:
            continue

        sorted_indices = torch.argsort(logits, descending=True).squeeze()
        try:
            rank = (sorted_indices == target_token_id).nonzero().item() + 1
            rank = min(rank, 1000)
            total_mrr += 1.0 / rank
            valid_token_count += 1

            predicted_token_id = torch.tensor([[target_token_id]]).to(device)
            current_input_ids = torch.cat([current_input_ids, predicted_token_id], dim=1)
        except:
            total_mrr += 1.0 / 1000
            warning_counter["Target token not found"] = warning_counter.get("Target token not found", 0) + 1

    return total_mrr / valid_token_count if valid_token_count > 0 else 0.0


def plot_mrr_distribution(metrics, threshold, save_path, privacy_type):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(metrics, bins=50, alpha=0.7, color='#1f77b4', label=f'All {privacy_type.capitalize()} Samples')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Memory Threshold={threshold}')
    high_memory_count = sum(1 for m in metrics if m > threshold)
    plt.xlabel('MRR (Mean Reciprocal Rank)', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'{privacy_type.capitalize()} MRR Distribution (Memory Samples: {high_memory_count}/{len(metrics)})',
              fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"MRR distribution plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_text_path",
                        default=r"D:\expt\lrp_spin\data_process\enron_txt\train.txt",
                        type=str,
                        help="Path to raw English email TXT file (Enron format)")
    parser.add_argument("--output_path",
                        default=r"D:\expt\lrp_spin\main\name\gpt2_memorized_privacy.txt",
                        type=str,
                        help="Output path for high-memory samples")
    parser.add_argument("--model_name_or_path",
                        default=r"D:\expt\lrp_spin\WT_model\2025_11_17_gpt2",
                        type=str,
                        help="Path to fine-tuned GPT2 model")
    parser.add_argument("--max_seq_length",
                        default=1024,
                        type=int,
                        help="GPT2 maximum sequence length")
    parser.add_argument("--privacy_type",
                        default='name',
                        choices=['name', 'phone'],
                        help="Privacy type: name (full name)/ phone (phone number)")
    parser.add_argument("--min_context_len",
                        type=int,
                        default=10,
                        help="Minimum context length (by word count)")
    parser.add_argument("--max_context_len",
                        type=int,
                        default=20,
                        help="Maximum context length (by word count)")
    parser.add_argument("--threshold",
                        type=float,
                        default=0.4,
                        help="Threshold for filtering high-memory samples (MRR>threshold)")
    parser.add_argument("--max_data_count",
                        type=int,
                        default=30000,
                        help="Maximum number of privacy samples to extract")
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
    parser.add_argument("--batch_size_stat",
                        type=int,
                        default=1000,
                        help="Statistics progress every N samples")
    parser.add_argument("--plot_dist",
                        default=True,
                        action='store_true',
                        help="Plot MRR distribution histogram")

    args = parser.parse_args()

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
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
        model.to(device)
        model.eval()
        torch.cuda.empty_cache()
        model_load_time = time.time() - model_load_start
        logger.info(
            f"Model loading completed | Time consumed: {model_load_time:.2f}s | "
            f"Hidden layer dimension: {model.config.n_embd} | Sequence length: {model.config.n_ctx}"
        )
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}", exc_info=True)
        return

    try:
        privacies, extract_time, total_raw_data = get_privacy_from_raw_text(
            args.raw_text_path,
            args.privacy_type,
            min_context_len=args.min_context_len,
            max_context_len=args.max_context_len,
            max_data_count=args.max_data_count
        )
        if not privacies:
            logger.warning(f"No valid {args.privacy_type} privacy data extracted, program exits")
            return
    except Exception as e:
        logger.error(f"Privacy data extraction failed: {str(e)}", exc_info=True)
        return

    memorized_text = []
    all_mrr = []
    avg_high_mrr = 0.0
    quant_start_time = time.time()
    total_privacys = len(privacies)
    batch_size = args.batch_size_stat
    batch_mrr = []
    batch_high_count = 0

    logger.info(
        f"\nStart calculating MRR (adapted for GPT2 autoregressive generation) | Total samples: {total_privacys} | "
        f"Statistics batch: every {batch_size} samples | Filter threshold: {args.threshold}"
    )

    with tqdm(total=total_privacys, desc=f"Quantifying {args.privacy_type} memory", unit="samples", ncols=100) as pbar:
        for idx, record in enumerate(privacies):
            try:
                context = record["context"]
                target_privacy = record["privacy"]
                privacy_mrr = calculate_privacy_mrr(
                    context=context,
                    target_privacy=target_privacy,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_seq_length=args.max_seq_length,
                    privacy_type=args.privacy_type
                )
                all_mrr.append(privacy_mrr)
                batch_mrr.append(privacy_mrr)

                if privacy_mrr > args.threshold:
                    line = f"context: {context} | privacy: {target_privacy} | MRR: {privacy_mrr:.4f}"
                    memorized_text.append(line)
                    avg_high_mrr += privacy_mrr
                    batch_high_count += 1

                pbar.update(1)

                if (idx + 1) % batch_size == 0 or (idx + 1) == total_privacys:
                    valid_batch_mrr = [m for m in batch_mrr if m > 1e-6]
                    batch_avg_mrr = sum(valid_batch_mrr) / len(valid_batch_mrr) if valid_batch_mrr else 0.0
                    start_idx = idx - len(batch_mrr) + 1
                    end_idx = idx + 1
                    logger.info(
                        f"Batch statistics [{start_idx + 1}-{end_idx}/{total_privacys}] | "
                        f"Average MRR: {batch_avg_mrr:.4f} | "
                        f"High-memory samples count: {batch_high_count} | "
                        f"Batch ratio: {batch_high_count / len(batch_mrr) * 100:.2f}%"
                    )
                    batch_mrr = []
                    batch_high_count = 0

            except Exception as e:
                warning_key = "MRR calculation error"
                warning_counter[warning_key] = warning_counter.get(warning_key, 0) + 1
                if warning_counter[warning_key] == 1:
                    logger.warning(f"Error processing sample {idx}: {str(e)[:50]} (First occurrence, only count later)")
                batch_mrr.append(0.0)
                all_mrr.append(0.0)
                pbar.update(1)

    quant_time = time.time() - quant_start_time

    if memorized_text:
        memorized_text.sort(key=lambda x: float(x.split('|')[-1].split(':')[-1].strip()), reverse=True)
        avg_high_mrr /= len(memorized_text)
        with open(args.output_path, "w", encoding="utf-8") as f:
            for line in memorized_text:
                f.write(line + "\n")
        total_time = time.time() - start_total_time

        logger.info("\n" + "=" * 80)
        logger.info(f"GPT2 {args.privacy_type} high-memory samples extraction completed (context optimization disabled)")
        logger.info(f"{'=' * 20} Time Statistics {'=' * 20}")
        logger.info(f"Model loading time: {model_load_time:.2f}s")
        logger.info(f"Privacy extraction time: {extract_time:.2f}s")
        logger.info(f"MRR calculation time: {quant_time:.2f}s")
        logger.info(f"Total process time: {total_time:.2f}s")
        logger.info(f"{'=' * 20} Result Statistics {'=' * 20}")
        logger.info(f"Total raw messages: {total_raw_data}")
        logger.info(f"Total extracted privacy: {total_privacys}/{args.max_data_count}")
        logger.info(f"Total high-memory samples: {len(memorized_text)}")
        logger.info(f"High-memory ratio: {len(memorized_text) / total_privacys * 100:.2f}%")
        logger.info(f"Average MRR of high-memory samples: {avg_high_mrr:.4f}")
        logger.info(f"High-memory samples save path: {args.output_path}")
        logger.info("=" * 80)

        if args.plot_dist:
            plot_save_path = os.path.splitext(args.output_path)[0] + "_mrr_dist.png"
            plot_mrr_distribution(all_mrr, args.threshold, plot_save_path, args.privacy_type)
    else:
        logger.warning(f"\nNo high-memory {args.privacy_type} samples filtered out!")
        logger.warning(f"Suggestions: 1.Reduce MRR threshold (current {args.threshold}) 2.Increase raw text volume 3.Adjust context length range")

    logger.info("\n" + "=" * 50)
    logger.info("Warning statistics summary:")
    for warning, count in warning_counter.items():
        logger.info(f"- {warning}: {count} times")
    logger.info("=" * 50)
    logger.info("Program exited normally!")


if __name__ == "__main__":
    main()