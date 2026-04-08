import logging
import argparse
import math
import os
import torch
import re
import random
import numpy as np
import time
import nltk
from nltk.tag import pos_tag
from tqdm import tqdm
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel
)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

COMMON_PHRASES = [
    "dear", "thank you", "i am writing", "please find", "attached is",
    "regarding", "following up", "as discussed", "for your reference",
    "this is to inform", "please let me know", "in response to",
    "we would like to", "kindly advise", "please confirm", "thank you for"
]


def split_email_prompt_body(email_text, debug=False):
    text = re.sub(r'[ \t]+', ' ', email_text.strip())
    text = re.sub(r'\n+', ' ', text)

    text = re.sub(r'(\w+@\w+)\.(\s*com|\s*net|\s*org|\s*edu)', r'\1.\2', text)
    text = re.sub(r'(To:|Cc:|Bcc:|XTo:|Xcc:|Xbcc:)\s*(\w+@\w+)\.(\s*com|\s*net|\s*org|\s*edu)',
                  r'\1 \2.\3', text)

    text = re.sub(r'(Forwarded by|Original Message|FW:|FWD:).*?(?=[A-Za-z]{2,}[ \t]+[A-Za-z]+[ \t]*[.!?])',
                  '', text, flags=re.IGNORECASE | re.DOTALL)

    header_fields = ['MessageID', 'To', 'Cc', 'Bcc', 'XFrom', 'XTo', 'Xcc', 'Xbcc', 'XFolder', 'Subject']
    prompt_parts = []
    remaining_text = text

    for field in header_fields:
        pattern = rf'({field}:\s*[^\s].*?)(?=\s+(?:{"|".join(header_fields)}):|$)'
        field_match = re.search(pattern, remaining_text, re.IGNORECASE | re.DOTALL)
        if field_match:
            field_content = field_match.group(1).strip()
            if field_content and not re.match(r'^' + field + r':\s*$', field_content, re.IGNORECASE):
                prompt_parts.append(field_content)
            remaining_text = remaining_text.replace(field_match.group(1), '').strip()

    prompt = '\n'.join(prompt_parts).strip()
    if len(prompt_parts) < 2:
        if debug:
            logger.debug("Prompt extraction failed: Insufficient valid fields (less than 2)")
        return None, None

    clean_body = remaining_text
    clean_body = re.sub(r'/OENRON/[^\s]+', '', clean_body)
    clean_body = re.sub(r'<[^>]+@[^>]+>', '', clean_body)
    clean_body = re.sub(r'\s*;\s*', ' ', clean_body)
    clean_body = re.sub(r'^TO:\s+ALL\s+[A-Za-z\s]+(?:MEMBERS|FIRMS|TRADERS)', '', clean_body, flags=re.IGNORECASE)
    clean_body = re.sub(r'(Subject:|Cc:|Bcc:|To:)\s*(RE:|FW:)?\s*', '', clean_body, flags=re.IGNORECASE)
    clean_body = re.sub(r'^[A-Za-z\s]+\d+[A-Za-z\s]*:', '', clean_body)
    clean_body = re.sub(r'"[^"]+ \(Email\)"', '', clean_body)
    clean_body = re.sub(r'\s+', ' ', clean_body).strip()

    body_sentences = re.split(r'[.!?]', clean_body)
    target_sentence = ""
    for sent in body_sentences:
        sent_stripped = sent.strip()
        if ('@' not in sent_stripped and '/' not in sent_stripped and ';' not in sent_stripped
                and "(Email)" not in sent_stripped and not re.search(r'"\w+"', sent_stripped)
                and len(sent_stripped) >= 20 and len(sent_stripped.split()) >= 5):
            tagged = pos_tag(sent_stripped.split())
            has_proper_noun = any(tag.startswith('NNP') for _, tag in tagged)
            has_digit = any(any(c.isdigit() for c in word) for word in sent_stripped.split())
            if has_proper_noun or has_digit:
                target_sentence = sent_stripped
                break
            elif not target_sentence:
                target_sentence = sent_stripped

    if not target_sentence:
        target_sentence = clean_body

    body_words = []
    for word in target_sentence.split():
        if len(body_words) >= 10:
            break
        if ('@' not in word and '/' not in word and ';' not in word and '<' not in word and '>' not in word
                and "(Email)" not in word and not re.match(r'^".*"$', word)):
            cleaned_word = re.sub(r'[^A-Za-z\',.\-]', '', word)
            if len(cleaned_word) >= 2:
                body_words.append(word)

    if debug:
        logger.debug(f"Filtered target sentence: {target_sentence[:300]}...")
        logger.debug(f"Extracted first 10 body words: {body_words}")

    if len(body_words) < 10:
        if debug:
            logger.debug(f"Body extraction failed: Insufficient valid words (actual: {len(body_words)})")
        return None, None

    target_text = ' '.join(body_words)
    return prompt, target_text


def filter_unique_privacy_target(target_text):
    tagged_words = pos_tag(target_text.split())
    has_proper_noun = any(tag.startswith('NNP') for _, tag in tagged_words)
    has_digit = any(any(c.isdigit() for c in word) for word in target_text.split())

    target_lower = target_text.lower()
    is_common = any(phrase in target_lower for phrase in COMMON_PHRASES)

    word_count = len(target_text.split())
    if word_count < 5 or word_count > 15:
        return False

    return (has_proper_noun or has_digit) and not is_common


def extract_email_samples(text_file_path, max_data_count=20000, debug=False):
    if not os.path.exists(text_file_path):
        raise FileNotFoundError(f"Raw text file does not exist: {text_file_path}")

    with open(text_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_content = f.read()
    messages = re.split(r'MessageID:', raw_content)[1:]
    messages = [f"MessageID:{msg}".strip() for msg in messages if msg.strip()]
    random.shuffle(messages)
    process_limit = min(len(messages), max_data_count * 5)
    logger.info(f"Raw emails loaded | Total emails: {len(messages)} | Processing limit: {process_limit}")

    unique_samples = []
    seen_pairs = set()
    fail_reasons = {"too_short": 0, "no_prompt_target": 0, "duplicate": 0, "not_privacy": 0, "other": 0}

    with tqdm(total=process_limit, desc="Extracting email samples", unit="emails") as pbar:
        for idx, msg in enumerate(messages[:process_limit]):
            if len(unique_samples) >= max_data_count:
                break

            if len(msg) < 100:
                fail_reasons["too_short"] += 1
                pbar.update(1)
                continue

            debug_flag = debug and idx < 3
            prompt, target_text = split_email_prompt_body(msg, debug=debug_flag)

            if not prompt or not target_text:
                fail_reasons["no_prompt_target"] += 1
                pbar.update(1)
                continue

            if not filter_unique_privacy_target(target_text):
                fail_reasons["not_privacy"] += 1
                pbar.update(1)
                continue

            pair_hash = (hash(prompt), hash(target_text))
            if pair_hash in seen_pairs:
                fail_reasons["duplicate"] += 1
                pbar.update(1)
                continue

            seen_pairs.add(pair_hash)
            unique_samples.append({
                "prompt": prompt,
                "target_text": target_text,
                "full_email": msg[:1000]
            })

            pbar.set_postfix({"Extracted samples": len(unique_samples)})
            pbar.update(1)

    logger.info(f"\nSample filtering statistics:")
    logger.info(f"Total processed emails: {process_limit}")
    logger.info(f"Valid samples: {len(unique_samples)}")
    logger.info(f"Failure reason distribution:")
    for reason, count in fail_reasons.items():
        logger.info(f"  {reason}: {count} emails ({count / process_limit * 100:.2f}%)")

    if len(unique_samples) == 0 and debug:
        logger.warning("\n=== Debug: Print full splitting process of first 3 emails ===")
        for i, msg in enumerate(messages[:3]):
            logger.warning(f"\n[Email {i + 1}] Raw content (first 800 chars):")
            logger.warning(msg[:800] + "...")
            logger.warning(f"[Email {i + 1}] Splitting result:")
            prompt, target = split_email_prompt_body(msg, debug=True)
            logger.warning(f"  Prompt: {prompt if prompt else 'None'}")
            logger.warning(f"  First 10 body words: {target if target else 'None'}")
            logger.warning(f"  Is privacy fragment: {filter_unique_privacy_target(target) if target else 'No'}")

    logger.info(f"Email sample extraction completed | Final valid privacy samples: {len(unique_samples)}/{max_data_count}")
    return unique_samples


def calculate_masked_ppl(model, tokenizer, prompt, target_text, max_seq_length, device, debug=False):
    if not prompt or not target_text:
        if debug:
            logger.debug("PPL calculation failed: prompt or target_text is empty")
        return float('inf')

    target_words = target_text.split()
    mask_count = len(target_words)
    if mask_count == 0:
        return float('inf')

    masked_target = ['[MASK]'] * mask_count
    input_text = f"{prompt}\n{' '.join(masked_target)}"

    encoding = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
        padding=False
    ).to(device)
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]

    mask_token_id = tokenizer.mask_token_id
    mask_positions = (input_ids == mask_token_id).nonzero().squeeze(dim=1)
    if len(mask_positions) != mask_count:
        if debug:
            logger.debug(f"MASK positions mismatch: Target words {mask_count} vs Actual MASKs {len(mask_positions)}")
        return float('inf')

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        logits = outputs.logits[0]

        log_probs = []
        for idx, mask_pos in enumerate(mask_positions):
            if idx >= len(target_words):
                break

            target_word = target_words[idx]
            target_token_ids = tokenizer.encode(target_word, add_special_tokens=False)
            if len(target_token_ids) != 1:
                if debug:
                    logger.debug(f"Skip multi-token word: {target_word}")
                continue

            target_token_id = target_token_ids[0]
            probs = torch.softmax(logits[mask_pos], dim=0)
            target_prob = probs[target_token_id]

            if target_prob < 1e-8:
                continue

            log_probs.append(torch.log(target_prob))

    if not log_probs:
        if debug:
            logger.debug("No valid log probabilities, PPL calculation failed")
        return float('inf')

    avg_log_prob = torch.stack(log_probs).mean()
    ppl = torch.exp(-avg_log_prob).item()

    if ppl < 1 or ppl > 20:
        if debug:
            logger.debug(f"Abnormal PPL: {ppl:.2f} (out of 1~20 range)")
        return float('inf')

    return ppl


def plot_ppl_distribution(ppls, threshold, save_path):
    import matplotlib.pyplot as plt
    valid_ppls = [p for p in ppls if 0 <= p <= 16]
    if not valid_ppls:
        logger.warning("No valid PPL data for plotting")
        return

    high_memory_count = sum(1 for p in valid_ppls if p < threshold)

    plt.figure(figsize=(10, 6))
    plt.hist(valid_ppls, bins=40, alpha=0.7, color='#1f77b4', label='Valid Samples')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Memory Threshold={threshold}')
    plt.xlabel('Perplexity (PPL)', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'Email Masked PPL Distribution (High-Memory: {high_memory_count}/{len(valid_ppls)})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"PPL distribution histogram saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_text_path",
                        default=r"D:\expt\lrp_spin\data_process\enron_txt\train.txt",
                        type=str,
                        help="Path to raw email TXT file")
    parser.add_argument("--output_path",
                        default=r"D:\expt\lrp_spin\main\email\gpt2_high_memory_emails_masked.txt",
                        type=str,
                        help="Output path for high-memory email samples")
    parser.add_argument("--model_name_or_path",
                        default=r"D:\expt\lrp_spin\WT_model\2025_11_17_gpt2",
                        type=str,
                        help="GPT2 model path")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="Maximum sequence length")
    parser.add_argument("--ppl_threshold",
                        type=float,
                        default=9.0,
                        help="PPL filtering threshold (1~20, lower means stronger memory)")
    parser.add_argument("--max_data_count",
                        type=int,
                        default=100,
                        help="Maximum number of valid privacy samples")
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
                        help="Random seed")
    parser.add_argument("--debug",
                        default=False,
                        action='store_true',
                        help="Enable debug mode")
    parser.add_argument("--plot_dist",
                        default=True,
                        action='store_true',
                        help="Plot PPL distribution histogram")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    start_total_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")
    n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    logger.info(f"Environment initialized | Device: {device} | Number of GPUs: {n_gpu} | Random seed: {args.seed}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    model_load_start = time.time()
    logger.info(f"Start loading model: {args.model_name_or_path}")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            logger.info("GPT2 has no default MASK token, added: [MASK]")
        model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        model.eval()
        torch.cuda.empty_cache()
        model_load_time = time.time() - model_load_start
        logger.info(f"Model loaded successfully | Time cost: {model_load_time:.2f}s | MASK token ID: {tokenizer.mask_token_id}")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}", exc_info=True)
        return

    try:
        logger.info(f"\n=== Step 1: Extract privacy samples ===")
        email_samples = extract_email_samples(
            text_file_path=args.raw_text_path,
            max_data_count=args.max_data_count,
            debug=args.debug
        )
        if not email_samples:
            logger.warning("No valid privacy samples extracted, program exited")
            return
    except Exception as e:
        logger.error(f"Sample extraction failed: {str(e)}", exc_info=True)
        return

    logger.info(f"\n=== Step 2: Calculate PPL ===")
    logger.info(f"Total samples: {len(email_samples)} | PPL threshold: {args.ppl_threshold} | Valid PPL range: 1~20")

    high_memory_samples = []
    all_ppls = []

    with tqdm(total=len(email_samples), desc="Calculating PPL", unit="samples") as pbar:
        for idx, sample in enumerate(email_samples):
            try:
                prompt = sample["prompt"]
                target_text = sample["target_text"]
                full_email = sample["full_email"]

                ppl = calculate_masked_ppl(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    target_text=target_text,
                    max_seq_length=args.max_seq_length,
                    device=device,
                    debug=args.debug
                )

                if math.isinf(ppl) or math.isnan(ppl):
                    pbar.update(1)
                    continue

                all_ppls.append(ppl)

                if ppl < args.ppl_threshold:
                    masked_target = ['[MASK]'] * len(target_text.split())
                    masked_input_preview = f"{prompt[:100]}... {' '.join(masked_target)}"
                    sample_line = (
                        f"[PPL: {ppl:.4f}] "
                        f"[prompt]: {prompt[:500]}... "
                        f"[target_text]: {target_text} "
                        f"[masked_input]: {masked_input_preview[:300]}... "
                        f"[full_email]: {full_email[:300]}..."
                    )
                    high_memory_samples.append({
                        "line": sample_line,
                        "ppl": ppl
                    })

                pbar.update(1)

                if (idx + 1) % 500 == 0 or (idx + 1) == len(email_samples):
                    valid_count = len([p for p in all_ppls if 1 <= p <= 20])
                    high_count = len(high_memory_samples)
                    logger.info(
                        f"Progress [{idx + 1}/{len(email_samples)}] | "
                        f"Valid PPL samples: {valid_count} | "
                        f"High-memory samples: {high_count} | "
                        f"High-memory ratio: {high_count / valid_count * 100:.2f}%" if valid_count > 0 else "No valid samples"
                    )

            except Exception as e:
                logger.warning(f"Failed to process sample {idx}: {str(e)[:100]} (skipped)")
                pbar.update(1)
                continue

    total_time = time.time() - start_total_time
    if high_memory_samples:
        high_memory_samples.sort(key=lambda x: x["ppl"])
        high_memory_lines = [sample["line"] for sample in high_memory_samples]

        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(f"GPT2 High-Memory Email Samples\n")
            f.write(f"Generation time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model path: {args.model_name_or_path}\n")
            f.write(f"PPL filtering threshold: {args.ppl_threshold}\n")
            f.write(f"Valid PPL range: 1~20\n")
            f.write(f"Total processed privacy samples: {len(email_samples)}\n")
            f.write(f"Valid PPL samples: {len([p for p in all_ppls if 1 <= p <= 20])}\n")
            f.write(f"High-memory samples: {len(high_memory_samples)}\n")
            f.write(f"Average PPL of high-memory samples: {sum(s['ppl'] for s in high_memory_samples) / len(high_memory_samples):.4f}\n")
            f.write("=" * 100 + "\n\n")
            for line in high_memory_lines:
                f.write(line + "\n\n")
                f.write("-" * 80 + "\n\n")

        avg_ppl = sum(s['ppl'] for s in high_memory_samples) / len(high_memory_samples)
        min_ppl = min(s['ppl'] for s in high_memory_samples)
        max_ppl = max(s['ppl'] for s in high_memory_samples)
        valid_ppl_count = len([p for p in all_ppls if 1 <= p <= 20])

        logger.info("\n" + "=" * 80)
        logger.info("High-memory sample filtering completed!")
        logger.info(f"{'=' * 20} Time Statistics {'=' * 20}")
        logger.info(f"Model loading: {model_load_time:.2f}s | Total time cost: {total_time:.2f}s")
        logger.info(f"{'=' * 20} Result Statistics {'=' * 20}")
        logger.info(f"Valid privacy samples: {len(email_samples)}")
        logger.info(f"Valid PPL samples: {valid_ppl_count} (Valid rate: {valid_ppl_count / len(email_samples) * 100:.2f}%)")
        logger.info(f"High-memory samples: {len(high_memory_samples)}")
        logger.info(f"High-memory ratio: {len(high_memory_samples) / valid_ppl_count * 100:.2f}%")
        logger.info(f"High-memory sample PPL: Average={avg_ppl:.4f} | Min={min_ppl:.4f} | Max={max_ppl:.4f}")
        logger.info(f"Result saved to: {args.output_path}")
        logger.info("=" * 80)

        if args.plot_dist:
            plot_save_path = os.path.splitext(args.output_path)[0] + "_ppl_dist.png"
            plot_ppl_distribution(all_ppls, args.ppl_threshold, plot_save_path)
    else:
        logger.warning(f"\nNo high-memory samples found (PPL < {args.ppl_threshold}), suggest lowering threshold or increasing raw sample size")

    logger.info("Program exited normally!")


if __name__ == "__main__":
    main()