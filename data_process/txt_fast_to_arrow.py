# -*- coding: utf-8 -*-
import os
import time
import multiprocessing
import warnings
from tqdm import tqdm
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets
from transformers import AutoTokenizer, logging as hf_logging

TXT_FILES = [r"D:\expt\lrp_spin\data_process\enron_txt\train.txt"]
TOKENIZER_PATH = r"D:\expt\lrp_spin\base_model\2025_11_10_gpt2"
BLOCK_SIZE = 512
OUTPUT_ROOT = r"D:\expt\lrp_spin\data_process\enron_arrow"
BATCH_SIZE = 256
NUM_PROC = 10
CHUNK_SIZE = 10000
LOG_INTERVAL = 10

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def load_tokenizer(tokenizer_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Automatically set pad_token to eos_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        else:
            print(f"pad_token already exists: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

        test_text = "Hello, this is a test for tokenizer."
        test_token = tokenizer(
            test_text,
            max_length=BLOCK_SIZE,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )
        print(f"Tokenizer test normal: Input text length {len(test_text)} → token count {len(test_token['input_ids'])}")
        print(f"  First 5 input_ids of test sample: {test_token['input_ids'][:5]}")
        print(f"  First 5 attention_mask of test sample: {test_token['attention_mask'][:5]}")
        return tokenizer
    except Exception as e:
        print(f"Tokenizer loading failed: {e}")
        exit(1)


def read_large_txt(file_path, batch_size=1000):
    batch_texts = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line_strip = line.strip()
            if line_strip and not line_strip.isspace():
                batch_texts.append(line_strip)
            if len(batch_texts) >= batch_size:
                yield batch_texts
                batch_texts = []
    if batch_texts:
        yield batch_texts


def process_batch(args):
    with open(os.devnull, 'w') as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)

    batch_texts, tokenizer_path, block_size = args
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    try:
        tokenized = tokenizer(
            batch_texts,
            padding=False,
            truncation=True,
            max_length=block_size,
            return_tensors=None,
            add_special_tokens=True
        )

        if len(tokenized["input_ids"]) != len(batch_texts):
            print(f"Batch processing exception: Input {len(batch_texts)} items → Output {len(tokenized['input_ids'])} items")
            return None

        input_ids, attention_mask, labels = [], [], []
        for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
            current_len = len(ids)
            pad_length = block_size - current_len

            if pad_length > 0:
                ids += [pad_token_id] * pad_length
                mask += [0] * pad_length
            elif pad_length < 0:
                ids = ids[:block_size]
                mask = mask[:block_size]

            label = [ids[i] if mask[i] == 1 else -100 for i in range(block_size)]

            assert len(ids) == block_size, f"input_ids length {len(ids)}≠{block_size}"
            assert len(mask) == block_size, f"attention_mask length {len(mask)}≠{block_size}"
            assert len(label) == block_size, f"labels length {len(label)}≠{block_size}"

            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(label)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return None


def save_chunk(chunk_data, save_dir, chunk_idx, total_chunks):
    if not chunk_data["input_ids"]:
        print(f"Chunk {chunk_idx} has no valid samples, skip saving")
        return

    features = Features({
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
    })

    dataset = Dataset.from_dict(chunk_data, features=features)
    chunk_path = os.path.join(save_dir, f"chunk_{chunk_idx}")
    os.makedirs(chunk_path, exist_ok=True)
    dataset.save_to_disk(chunk_path)

    if (chunk_idx + 1) % LOG_INTERVAL == 0 or chunk_idx + 1 == total_chunks:
        total_samples = (chunk_idx + 1) * CHUNK_SIZE
        print(f"Saved {chunk_idx + 1}/{total_chunks} chunks → Total samples: {total_samples:,}")


def verify_dataset_quality(final_path):
    print(f"\nStart verifying dataset quality: {final_path}")
    dataset = Dataset.load_from_disk(final_path)
    sample = dataset[0]

    required_fields = ["input_ids", "attention_mask", "labels"]
    for field in required_fields:
        assert field in dataset.column_names, f"Missing required field: {field}"
    print("Required fields verification passed")

    assert len(sample["input_ids"]) == BLOCK_SIZE, f"input_ids length {len(sample['input_ids'])}≠{BLOCK_SIZE}"
    assert len(sample["attention_mask"]) == BLOCK_SIZE, f"attention_mask length mismatch"
    assert len(sample["labels"]) == BLOCK_SIZE, f"labels length mismatch"
    print("Sequence length verification passed")

    mask_values = set(sample["attention_mask"])
    assert 0 in mask_values, "attention_mask is all 1, pad positions not marked correctly"
    assert 1 in mask_values, "attention_mask is all 0, no valid text"
    print("attention_mask format verification passed")

    label_values = set(sample["labels"])
    assert -100 in label_values, "labels do not contain -100, cannot ignore pad position loss"
    print("labels format verification passed")

    valid_mask = [1 if m == 1 else 0 for m in sample["attention_mask"]]
    valid_labels = [l for l, m in zip(sample["labels"], sample["attention_mask"]) if m == 1]
    invalid_labels = [l for l, m in zip(sample["labels"], sample["attention_mask"]) if m == 0]
    assert all(l != -100 for l in valid_labels), "labels of valid text positions are marked as -100"
    assert all(l == -100 for l in invalid_labels), "labels of pad positions are not marked as -100"
    print("Corresponding relationship between labels and attention_mask verification passed")

    print(f"\nExample of last 10 values of sample:")
    print(f"Last 10 input_ids: {sample['input_ids'][-10:]}")
    print(f"Last 10 attention_mask: {sample['attention_mask'][-10:]}")
    print(f"Last 10 labels: {sample['labels'][-10:]}")
    print(f"All dataset quality verifications passed!")


def convert_txt_to_arrow(raw_txt_path, save_dir, tokenizer_path, batch_size, num_proc, block_size, chunk_size):
    if not os.path.exists(raw_txt_path):
        print(f"Input file does not exist: {raw_txt_path}")
        return None

    print(f"Reading text file: {raw_txt_path}")
    text_batches = list(read_large_txt(raw_txt_path, batch_size=batch_size))
    if not text_batches:
        print(f"No valid text data: {raw_txt_path}")
        return None
    print(f"Split into {len(text_batches)} batches, {batch_size} texts per batch")

    print(f"Start multi-process processing ({num_proc} processes)")
    valid_tokenized = []
    tasks = [(batch, tokenizer_path, block_size) for batch in text_batches]

    with multiprocessing.Pool(processes=num_proc) as pool:
        for batch_result in tqdm(
                pool.imap(process_batch, tasks),
                total=len(text_batches),
                desc="Processing progress"
        ):
            if batch_result is not None:
                valid_tokenized.append(batch_result)

    if not valid_tokenized:
        print(f"All batches processing failed: {raw_txt_path}")
        return None
    print(f"Successfully processed {len(valid_tokenized)}/{len(text_batches)} batches")

    print(f"Merging batches and saving chunks ({chunk_size} samples per chunk)")
    merged = {"input_ids": [], "attention_mask": [], "labels": []}
    chunk_idx = 0
    total_samples = sum(len(batch["input_ids"]) for batch in valid_tokenized)
    total_chunks = (total_samples + chunk_size - 1) // chunk_size

    for batch in valid_tokenized:
        merged["input_ids"].extend(batch["input_ids"])
        merged["attention_mask"].extend(batch["attention_mask"])
        merged["labels"].extend(batch["labels"])

        if len(merged["input_ids"]) >= chunk_size:
            save_chunk(merged, save_dir, chunk_idx, total_chunks)
            merged = {"input_ids": [], "attention_mask": [], "labels": []}
            chunk_idx += 1

    if merged["input_ids"]:
        save_chunk(merged, save_dir, chunk_idx, total_chunks)
        chunk_idx += 1

    print(f"Merging {chunk_idx} chunks into final dataset")
    all_chunks = []
    for i in range(chunk_idx):
        chunk_path = os.path.join(save_dir, f"chunk_{i}")
        all_chunks.append(Dataset.load_from_disk(chunk_path))

    final_dataset = concatenate_datasets(all_chunks)
    final_path = os.path.join(save_dir, "final")
    final_dataset.save_to_disk(final_path)
    print(f"Final dataset saved successfully!")
    print(f"  Save path: {final_path}")
    print(f"  Total samples: {len(final_dataset):,}")

    verify_dataset_quality(final_path)

    return final_path


if __name__ == "__main__":
    multiprocessing.freeze_support()
    total_start = time.time()

    tokenizer = load_tokenizer(TOKENIZER_PATH)

    for txt_path in TXT_FILES:
        print(f"\n" + "=" * 80)
        print(f"Start processing file: {txt_path}")
        print("=" * 80)

        txt_filename = os.path.splitext(os.path.basename(txt_path))[0]
        save_dir = os.path.join(OUTPUT_ROOT, txt_filename)
        os.makedirs(save_dir, exist_ok=True)

        final_path = convert_txt_to_arrow(
            raw_txt_path=txt_path,
            save_dir=save_dir,
            tokenizer_path=TOKENIZER_PATH,
            batch_size=BATCH_SIZE,
            num_proc=NUM_PROC,
            block_size=BLOCK_SIZE,
            chunk_size=CHUNK_SIZE
        )

    total_time = time.time() - total_start
    print(f"\nAll files processed! Total time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"Result root directory: {os.path.abspath(OUTPUT_ROOT)}")