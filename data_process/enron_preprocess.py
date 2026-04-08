# -*- coding: utf-8 -*-
import os
import re
import random
import time
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

MAIL_DIR = r'D:\expt\lrp_spin\data_process\maildir'
OUTPUT_TEL_PATH = r"D:\expt\lrp_spin\data_process\enron_txt\all_Tel.txt"
OUTPUT_DATA_DIR = r"D:\expt\lrp_spin\data_process\enron_txt"

EMAIL_REGEX = re.compile(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+")
NUMBER_REGEX = re.compile(r"\d+")
MULTI_SPACE_REGEX = re.compile(r'\s+')

PHONE_PROMPTS = [
    'My phone number is ***. ', 'You can call me at ***. ', 'Dial *** to call me. ',
    'Contact me at ***. ', 'Tel : ***. ', 'My fax number is ***. ',
    'Fax : ***. ', 'You can send a fax to ***. ', 'I can receive faxes at ***. ',
    '*** is the fax number to reach me. '
]

MAX_WORKERS = min(multiprocessing.cpu_count() - 4, 20)


def refilter_email(text):
    emails = EMAIL_REGEX.findall(text.lower())
    for email in emails:
        text = text.replace(f"{email},", "")
    return text


def add_prompted_nums(numbers):
    random_prompt = random.choice(PHONE_PROMPTS)
    return random_prompt.replace('***', ' '.join(list(numbers)))


def process_single_file(file_path):
    file_tel_list = []
    processed_text = ""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
        for line in all_lines:
            line_strip = line.strip()
            if line_strip.startswith((
                'X-FileName:', 'MessageID', 'Date: ', 'From: ', 'Subject: ',
                'Mime-Version: ', 'Content-Type: ', 'Content-Transfer-Encoding: ',
                'X-cc: ', 'X-bcc: ', 'X-Origin'
            )):
                continue
            cleaned_line = line_strip.replace('\\n', ' ').replace('-', '').replace('*', '') \
                .replace('#', '').replace('_', '').replace('=', '') + ' '
            numbers = ''.join(NUMBER_REGEX.findall(cleaned_line))
            if len(numbers) == 10:
                prompted_num = add_prompted_nums(numbers)
                cleaned_line = prompted_num
                file_tel_list.append(prompted_num)
            cleaned_line = MULTI_SPACE_REGEX.sub(' ', cleaned_line)
            cleaned_line = refilter_email(cleaned_line)
            processed_text += cleaned_line + ' '
        if len(processed_text.split()) >= 5:
            return (processed_text.lstrip() + '\n', file_tel_list)
        else:
            return (None, file_tel_list)
    except Exception:
        return (None, [])


def split_and_save_data(all_valid_texts, all_tels):
    if not os.path.exists(OUTPUT_DATA_DIR):
        os.makedirs(OUTPUT_DATA_DIR)
    unique_tels = list(set(all_tels))
    with open(OUTPUT_TEL_PATH, 'w', encoding='utf-8') as f:
        tel_lines = [tel.replace('.', '') + '\n' for tel in unique_tels]
        f.writelines(tel_lines)
    total_count = len(all_valid_texts)
    valid_texts = all_valid_texts[-3000:] if total_count >= 3000 else []
    test_texts = all_valid_texts[-6000:-3000] if total_count >= 6000 else []
    train_texts = all_valid_texts[:-6000] if total_count >= 6000 else all_valid_texts
    with open(os.path.join(OUTPUT_DATA_DIR, "train.txt"), 'w', encoding='utf-8') as f:
        f.writelines(train_texts)
    with open(os.path.join(OUTPUT_DATA_DIR, "test.txt"), 'w', encoding='utf-8') as f:
        f.writelines(test_texts)
    with open(os.path.join(OUTPUT_DATA_DIR, "valid.txt"), 'w', encoding='utf-8') as f:
        f.writelines(valid_texts)
    print(f"\n✅ Data saved successfully:")
    print(f"- Phone numbers: {OUTPUT_TEL_PATH} (Total {len(unique_tels)})")
    print(f"- Dataset: {OUTPUT_DATA_DIR} (Train {len(train_texts)} / Test {len(test_texts)} / Valid {len(valid_texts)})")


if __name__ == '__main__':
    total_start_time = time.time()

    print("🔍 Scanning all email files...")
    file_list = []
    for root, dirs, files in os.walk(MAIL_DIR):
        for file_name in files:
            file_list.append(os.path.join(root, file_name))
    print(f"✅ Scanned {len(file_list)} email files in total")

    print(f"\n⚡ Processing files with multiprocessing ({MAX_WORKERS} processes in parallel)...")
    all_valid_texts = []
    all_tel_numbers = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = tqdm(executor.map(process_single_file, file_list), total=len(file_list))
        for text, tels in results:
            if text:
                all_valid_texts.append(text)
            if tels:
                all_tel_numbers.extend(tels)

    print(f"\n📤 Saving processing results...")
    split_and_save_data(all_valid_texts, all_tel_numbers)

    total_elapsed = time.time() - total_start_time
    print(f"\n🎉 All processing completed! Total time consumed: {total_elapsed:.0f} seconds (about {total_elapsed/60:.1f} minutes)")
    print(f"- Total number of valid texts: {len(all_valid_texts)}")
    print(f"- Total extracted phone numbers: {len(all_tel_numbers)} ( {len(set(all_tel_numbers))} after deduplication)")