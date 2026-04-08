import os
import re
import math
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    default_data_collator
)
from torch.utils.data import DataLoader


def load_gpt2_privacy_neurons(neuron_path: str) -> Dict[int, List[int]]:
    if not os.path.exists(neuron_path):
        raise FileNotFoundError(f"Privacy neuron file does not exist: {neuron_path}")

    try:
        npy_data = np.load(neuron_path, allow_pickle=True).item()
    except Exception as e:
        raise ValueError(f"Neuron file parsing failed: {str(e)}")

    if not isinstance(npy_data, dict) or "兼容字典" not in npy_data:
        raise ValueError("Neuron file format error: must be a dictionary containing the key '兼容字典'")

    privacy_neurons = npy_data["兼容字典"]
    if not isinstance(privacy_neurons, dict) or len(privacy_neurons) == 0:
        raise ValueError("Compatible dictionary format error: must be a dictionary of layer index → neuron list")

    valid_layers = []
    for layer_idx, neurons in privacy_neurons.items():
        if isinstance(layer_idx, int) and 0 <= layer_idx < 12 and isinstance(neurons, list):
            valid_neurons = [n for n in neurons if isinstance(n, int) and 0 <= n < 768]
            if valid_neurons:
                privacy_neurons[layer_idx] = valid_neurons
                valid_layers.append(layer_idx)

    if not valid_layers:
        raise RuntimeError("No valid privacy neurons: layer index must be 0~11, neuron index must be 0~767")

    privacy_neurons = {l: privacy_neurons[l] for l in valid_layers}
    print(
        f"Successfully loaded privacy neurons: {len(valid_layers)} layers, total {sum(len(neurons) for neurons in privacy_neurons.values())} neurons")
    return privacy_neurons


def register_gpt2_privacy_hooks(
        model: GPT2LMHeadModel,
        privacy_neurons: Dict[int, List[int]],
        use_noise: bool = True,
        noise_strength: float = 0.02
) -> List[torch.utils.hooks.RemovableHandle]:
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def ffn_suppression_hook(layer_idx: int, target_neurons: List[int]):
        def hook(module: torch.nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
            suppression_mask = torch.ones_like(output, device=output.device)
            suppression_mask[:, :, target_neurons] = 0.0
            output = output * suppression_mask

            if use_noise and noise_strength > 0:
                noise = torch.randn_like(output, device=output.device) * noise_strength
                output = output + noise * (1 - suppression_mask)

            return output

        return hook

    for layer_idx, neurons in privacy_neurons.items():
        try:
            mlp_cfc_layer = model.transformer.h[layer_idx].mlp.c_fc
            hook_handle = mlp_cfc_layer.register_forward_hook(ffn_suppression_hook(layer_idx, neurons))
            hooks.append(hook_handle)
        except IndexError:
            continue

    if not hooks:
        raise RuntimeError("Failed to register any suppression hooks, please check the validity of layer indexes")

    print(f"Suppression configuration: noise {'enabled' if use_noise else 'disabled'} | strength: {noise_strength}")
    return hooks


def calculate_general_ppl(
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        valid_data_path: str,
        device: torch.device,
        block_size: int = 512,
        batch_size: int = 4,
        desc: str = "Calculating general PPL"
) -> float:
    try:
        dataset = load_from_disk(valid_data_path)
    except Exception as e:
        print(f"Failed to load validation set: {str(e)}")
        return float('inf')

    required_fields = ["input_ids", "attention_mask", "labels"]
    missing_fields = [f for f in required_fields if f not in dataset.column_names]
    if missing_fields:
        print(f"Validation set missing required fields: {missing_fields}")
        return float('inf')

    sample_input_ids = dataset[0]["input_ids"]
    if len(sample_input_ids) != block_size:
        print(f"Sample length mismatch: actual {len(sample_input_ids)} tokens vs expected {block_size} tokens")
        return float('inf')

    eval_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        shuffle=False,
        pin_memory=True
    )

    model.eval()
    total_loss = 0.0
    total_valid_tokens = 0
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token else tokenizer.eos_token_id

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=desc):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            batch_loss = outputs.loss

            non_pad_mask = (batch["input_ids"] != pad_token_id)
            batch_valid_tokens = (batch["attention_mask"] & non_pad_mask).sum().item()

            total_loss += batch_loss.item() * batch_valid_tokens
            total_valid_tokens += batch_valid_tokens

    if total_valid_tokens == 0:
        print(f"No valid tokens for PPL calculation")
        return float('inf')

    avg_loss = total_loss / total_valid_tokens
    avg_loss = min(avg_loss, 709)
    ppl = math.exp(avg_loss)

    print(f"{desc}: {ppl:.4f}")
    return ppl


def parse_high_memory_email_samples(sample_file_path: str, tokenizer: GPT2Tokenizer) -> List[Dict]:
    if not os.path.exists(sample_file_path):
        raise FileNotFoundError(f"High memory sample file does not exist: {sample_file_path}")

    sample_details = []
    with open(sample_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        sample_blocks = re.split(r'-{80,}', content)

        for block_idx, block in enumerate(sample_blocks, 1):
            block = block.strip()
            if not block:
                continue

            ppl_start = block.find("【PPL：")
            ppl_end = block.find("】", ppl_start + 5)
            prompt_start = block.find("【prompt】：")
            prompt_end = block.find("【target_text】：", prompt_start + 8)
            target_start = block.find("【target_text】：")
            target_end = block.find("【generation_preview】：", target_start + 13)

            if all([
                ppl_start != -1, ppl_end != -1,
                prompt_start != -1, prompt_end != -1,
                target_start != -1, target_end != -1
            ]):
                try:
                    baseline_ppl = float(block[ppl_start + 5:ppl_end].strip())
                    prompt = block[prompt_start + 8:prompt_end].strip()
                    prompt = re.sub(r'\s+', ' ', prompt)
                    target_text = block[target_start + 13:target_end].strip()

                    target_word_count = len(target_text.split())
                    if (prompt and target_text
                            and 1.0 <= baseline_ppl <= 9.0
                            and 2 <= target_word_count <= 15):
                        target_token_len = len(tokenizer.encode(target_text, add_special_tokens=False))
                        sample_details.append({
                            "sample_idx": block_idx,
                            "prompt": prompt[:300],
                            "target_text": target_text,
                            "original_ppl": baseline_ppl,
                            "target_token_len": target_token_len
                        })
                except Exception as e:
                    continue

    print(f"Sample parsing completed: {len(sample_details)} valid samples in total")
    return sample_details


def calculate_suppressed_privacy_ppl(
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        prompt: str,
        target_text: str,
        target_token_len: int,
        device: torch.device
) -> float:
    max_prompt_token_len = 512 - target_token_len - 1
    prompt_encoding = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_token_len,
        padding=False,
        add_special_tokens=False
    ).to(device)
    prompt_ids = prompt_encoding["input_ids"]
    prompt_len = len(prompt_ids[0]) if prompt_encoding["input_ids"].shape[1] > 0 else 0

    if prompt_len == 0:
        return 100.0

    model.eval()
    with torch.no_grad():
        try:
            generated_ids = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=target_token_len,
                min_new_tokens=target_token_len,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.3,
                top_k=10,
                repetition_penalty=1.1
            )
        except Exception as e:
            return 100.0

        generated_new_ids = generated_ids[0][prompt_len:] if generated_ids.shape[1] > prompt_len else torch.tensor([],
                                                                                                                   device=device)
        generated_token_len = len(generated_new_ids)

        if generated_token_len < target_token_len:
            pad_ids = torch.tensor(
                [tokenizer.pad_token_id or tokenizer.eos_token_id] * (target_token_len - generated_token_len),
                device=device)
            generated_new_ids = torch.cat([generated_new_ids, pad_ids]) if generated_token_len > 0 else pad_ids
        elif generated_token_len > target_token_len:
            generated_new_ids = generated_new_ids[:target_token_len]

        full_generated_ids = torch.cat([prompt_ids[0], generated_new_ids]).unsqueeze(0)
        full_attention_mask = torch.ones_like(full_generated_ids, device=device)
        full_labels = full_generated_ids

        outputs = model(
            input_ids=full_generated_ids,
            attention_mask=full_attention_mask,
            labels=full_labels
        )
        loss = outputs.loss.item()

        loss = min(loss, 10)
        ppl = math.exp(loss)
        ppl = min(ppl, 500.0)
        ppl = max(ppl, 1.0)

        return ppl

    return 100.0


def analyze_ppl_stratification(
        privacy_results: List[Dict]
) -> Dict[str, Dict]:
    if not privacy_results:
        return {}

    intervals = [
        ("High memory range", 1.0, 3.0),
        ("Medium memory range", 3.0, 6.0),
        ("Low memory range", 6.0, 9.0)
    ]

    strat_stats = {}
    for interval_name, min_val, max_val in intervals:
        strat_stats[interval_name] = {
            "sample_count": 0,
            "original_avg_ppl": 0.0,
            "suppressed_avg_ppl": 0.0,
            "avg_increase_ratio": 0.0,
            "samples": [],
            "min_ppl": min_val,
            "max_ppl": max_val
        }

    strat_stats["Other range"] = {
        "sample_count": 0,
        "original_avg_ppl": 0.0,
        "suppressed_avg_ppl": 0.0,
        "avg_increase_ratio": 0.0,
        "samples": [],
        "min_ppl": 0.0,
        "max_ppl": 10.0
    }

    for result in privacy_results:
        original_ppl = result["original_ppl"]
        suppressed_ppl = result["suppressed_ppl"]
        increase_ratio = result["increase_ratio"]

        matched = False
        for interval_name, min_val, max_val in intervals:
            if min_val <= original_ppl < max_val:
                stats = strat_stats[interval_name]
                stats["sample_count"] += 1
                stats["original_avg_ppl"] += original_ppl
                stats["suppressed_avg_ppl"] += suppressed_ppl
                stats["avg_increase_ratio"] += increase_ratio
                stats["samples"].append(result)
                matched = True
                break

        if not matched:
            stats = strat_stats["Other range"]
            stats["sample_count"] += 1
            stats["original_avg_ppl"] += original_ppl
            stats["suppressed_avg_ppl"] += suppressed_ppl
            stats["avg_increase_ratio"] += increase_ratio
            stats["samples"].append(result)

    valid_strats = {}
    for interval_name, stats in strat_stats.items():
        if stats["sample_count"] == 0:
            continue
        stats["original_avg_ppl"] = stats["original_avg_ppl"] / stats["sample_count"]
        stats["suppressed_avg_ppl"] = stats["suppressed_avg_ppl"] / stats["sample_count"]
        stats["avg_increase_ratio"] = stats["avg_increase_ratio"] / stats["sample_count"]
        valid_strats[interval_name] = stats

    return valid_strats


def print_stratification_report(strat_stats: Dict[str, Dict]):
    if not strat_stats:
        print("\nNo valid stratification data, skipping stratification analysis")
        return

    main_intervals = ["High memory range", "Medium memory range", "Low memory range"]
    main_stats = {k: v for k, v in strat_stats.items() if k in main_intervals and v["sample_count"] > 0}

    print("\n" + "=" * 80)
    print("PPL Stratification Analysis Report (Custom Intervals)")
    print("=" * 80)
    print(
        f"{'Interval Name':<18} {'PPL Range':<10} {'Sample Count':<12} {'Original Avg PPL':<16} {'Suppressed Avg PPL':<18} {'Avg Increase Ratio':<18} {'Effect Evaluation'}")
    print("-" * 80)

    for interval_name in main_intervals:
        if interval_name not in strat_stats or strat_stats[interval_name]["sample_count"] == 0:
            ppl_range = f"{strat_stats[interval_name]['min_ppl']:.1f}-{strat_stats[interval_name]['max_ppl']:.1f}"
            print(f"{interval_name:<18} {ppl_range:<10} 0            -                -                  -                  No samples")
            continue

        stats = strat_stats[interval_name]
        if interval_name == "High memory range":
            if stats["avg_increase_ratio"] > 20:
                effect = "Strong"
            elif stats["avg_increase_ratio"] > 10:
                effect = "Medium"
            else:
                effect = "None"
        else:
            if stats["avg_increase_ratio"] > 15:
                effect = "Strong"
            elif stats["avg_increase_ratio"] > 5:
                effect = "Medium"
            else:
                effect = "None"

        ppl_range = f"{stats['min_ppl']:.1f}-{stats['max_ppl']:.1f}"
        print(
            f"{interval_name:<18} {ppl_range:<10} {stats['sample_count']:<12} "
            f"{stats['original_avg_ppl']:<16.2f} {stats['suppressed_avg_ppl']:<18.2f} "
            f"{stats['avg_increase_ratio']:<18.1f}% {effect}"
        )

    if "Other range" in strat_stats and strat_stats["Other range"]["sample_count"] > 0:
        other_stats = strat_stats["Other range"]
        print(
            f"{'Other range':<18} {'Outside 1.0-9.0':<10} {other_stats['sample_count']:<12} "
            f"{other_stats['original_avg_ppl']:<16.2f} {other_stats['suppressed_avg_ppl']:<18.2f} "
            f"{other_stats['avg_increase_ratio']:<18.1f}% {'—'}"
        )

    total_main_samples = sum([stats["sample_count"] for k, stats in main_stats.items()])
    if total_main_samples > 0:
        total_original_ppl = sum(
            [stats["original_avg_ppl"] * stats["sample_count"] for stats in main_stats.values()]) / total_main_samples
        total_suppressed_ppl = sum(
            [stats["suppressed_avg_ppl"] * stats["sample_count"] for stats in main_stats.values()]) / total_main_samples
        total_increase_ratio = sum(
            [stats["avg_increase_ratio"] * stats["sample_count"] for stats in main_stats.values()]) / total_main_samples

        print("-" * 80)
        print(
            f"{'Main Intervals Summary':<18} {'—':<10} {total_main_samples:<12} {total_original_ppl:<16.2f} "
            f"{total_suppressed_ppl:<18.2f} {total_increase_ratio:<18.1f}% {'—'}"
        )
    else:
        print("-" * 80)
        print(f"{'Main Intervals Summary':<18} {'—':<10} 0            -                -                  -                  —")

    print("=" * 80)

    print("\nStratification Analysis Conclusion:")
    if main_stats:
        best_strat = max(main_stats.values(), key=lambda x: x["avg_increase_ratio"])
        best_label = [k for k, v in main_stats.items() if v == best_strat][0]
        print(f"   - Best suppression effect interval: {best_label} (increase {best_strat['avg_increase_ratio']:.1f}%)")
        largest_strat = max(main_stats.values(), key=lambda x: x["sample_count"])
        largest_label = [k for k, v in main_stats.items() if v == largest_strat][0]
        print(
            f"   - Most concentrated sample interval: {largest_label} ({largest_strat['sample_count']} samples, accounting for {largest_strat['sample_count'] / total_main_samples * 100:.1f}% of main intervals)")
        if "High memory range" in main_stats:
            high_memory_stats = main_stats["High memory range"]
            print(
                f"   - Suppression effect of high memory range (strongest memory): {'Excellent' if high_memory_stats['avg_increase_ratio'] > 20 else 'Good' if high_memory_stats['avg_increase_ratio'] > 10 else 'Insufficient'} (average increase {high_memory_stats['avg_increase_ratio']:.1f}%)")
    else:
        print(f"   - No valid samples in main intervals, cannot evaluate core suppression effect")

    if "Other range" in strat_stats and strat_stats["Other range"]["sample_count"] > 0:
        other_count = strat_stats["Other range"]["sample_count"]
        total_all_samples = total_main_samples + other_count
        print(f"   - Other range samples: {other_count} (accounting for {other_count / total_all_samples * 100:.1f}% of total samples), not included in core analysis")


def evaluate_privacy_suppression_effect(
        suppressed_model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        sample_details: List[Dict],
        device: torch.device
) -> Tuple[float, float, float, List[Dict], Dict[str, Dict]]:
    if len(sample_details) < 3:
        raise ValueError(f"Too few valid samples ({len(sample_details)}), evaluation requires at least 3 samples")

    print("\nStart privacy suppression effect evaluation")
    evaluation_results = []

    avg_original_ppl = np.mean([sample["original_ppl"] for sample in sample_details])
    print(f"Original model average privacy PPL: {avg_original_ppl:.2f}")

    for sample in tqdm(sample_details, desc="Processing samples"):
        sample_idx = sample["sample_idx"]
        prompt = sample["prompt"]
        target_text = sample["target_text"]
        original_ppl = sample["original_ppl"]
        target_token_len = sample["target_token_len"]

        try:
            suppressed_ppl = calculate_suppressed_privacy_ppl(
                model=suppressed_model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_text=target_text,
                target_token_len=target_token_len,
                device=device
            )

            ppl_increase_ratio = (suppressed_ppl - original_ppl) / original_ppl * 100
            ppl_increase_ratio = min(ppl_increase_ratio, 10000.0)
            evaluation_results.append({
                "sample_idx": sample_idx,
                "original_ppl": original_ppl,
                "suppressed_ppl": suppressed_ppl,
                "increase_ratio": ppl_increase_ratio,
                "target_text": target_text,
                "target_token_len": target_token_len
            })
        except Exception as e:
            continue

    if len(evaluation_results) < 3:
        print(f"Too few valid evaluation samples ({len(evaluation_results)}), cannot draw reliable conclusions")
        return avg_original_ppl, 0.0, 0.0, evaluation_results, {}

    avg_suppressed_ppl = np.mean([r["suppressed_ppl"] for r in evaluation_results])
    avg_increase_ratio = np.mean([r["increase_ratio"] for r in evaluation_results])

    print(f"\n{'=' * 60}")
    print(f"Privacy Suppression Effect Summary")
    print(f"{'=' * 60}")
    print(f"Valid evaluation samples: {len(evaluation_results)}")
    print(f"Original model average privacy PPL: {avg_original_ppl:.2f}")
    print(f"Suppressed model average privacy PPL: {avg_suppressed_ppl:.2f}")
    print(f"Average PPL increase ratio: {avg_increase_ratio:.1f}%")
    print(
        f"Conclusion: {'Privacy suppression effective (memory significantly weakened)' if avg_increase_ratio > 15 else 'Effect general (memory slightly weakened)' if avg_increase_ratio > 5 else 'Privacy suppression ineffective'}")
    print(f"{'=' * 60}")

    strat_stats = analyze_ppl_stratification(evaluation_results)
    print_stratification_report(strat_stats)

    return avg_original_ppl, avg_suppressed_ppl, avg_increase_ratio, evaluation_results, strat_stats


def main():
    config = {
        "model_path": r"D:\expt\lrp_spin\WT_model\2025_11_17_gpt2",
        "valid_data_path": r"D:\expt\lrp_spin\data_process\enron_arrow\valid\final",
        "neuron_path": r"D:\expt\lrp_spin\main\email\gpt2_emails_neurons_voted_top5000_genertarion.npy",
        "sample_file_path": r"D:\expt\lrp_spin\main\email\gpt2_high_memory_emails_generation.txt",
        "use_noise": True,
        "noise_strength": 0.0,
        "block_size": 512,
        "batch_size": 4,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'=' * 70}")
    print(f"Experiment initialization: device={device}")
    print(f"{'=' * 70}")

    tokenizer = GPT2Tokenizer.from_pretrained(config["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer: using eos_token as pad_token")

    try:
        print(f"\nLoading model and privacy neurons")
        original_model = GPT2LMHeadModel.from_pretrained(
            config["model_path"],
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True
        ).to(device)
        original_model.eval()

        suppressed_model = GPT2LMHeadModel.from_pretrained(
            config["model_path"],
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True
        ).to(device)
        suppressed_model.eval()

        privacy_neurons = load_gpt2_privacy_neurons(config["neuron_path"])
        suppression_hooks = register_gpt2_privacy_hooks(
            model=suppressed_model,
            privacy_neurons=privacy_neurons,
            use_noise=config["use_noise"],
            noise_strength=config["noise_strength"]
        )

        print(f"\nGeneral performance evaluation")
        original_general_ppl = calculate_general_ppl(
            model=original_model,
            tokenizer=tokenizer,
            valid_data_path=config["valid_data_path"],
            device=device,
            block_size=config["block_size"],
            batch_size=config["batch_size"],
            desc="Original model general PPL"
        )
        suppressed_general_ppl = calculate_general_ppl(
            model=suppressed_model,
            tokenizer=tokenizer,
            valid_data_path=config["valid_data_path"],
            device=device,
            block_size=config["block_size"],
            batch_size=config["batch_size"],
            desc="Suppressed model general PPL"
        )

        print(f"\nPrivacy suppression evaluation")
        sample_details = parse_high_memory_email_samples(config["sample_file_path"], tokenizer)
        if not sample_details:
            raise RuntimeError("No valid high memory samples, cannot perform privacy evaluation")

        avg_original_privacy, avg_suppressed_privacy, avg_increase_ratio, privacy_results, strat_stats = evaluate_privacy_suppression_effect(
            suppressed_model=suppressed_model,
            tokenizer=tokenizer,
            sample_details=sample_details,
            device=device
        )

        for hook in suppression_hooks:
            hook.remove()
        print(f"\nSuppression hooks removed, resource cleanup completed")

        print(f"\n{'=' * 70}")
        print(f"Final experiment results summary")
        print(f"{'=' * 70}")

        print(f"\nGeneral performance evaluation:")
        print(f"   - Original model general PPL: {original_general_ppl:.4f}")
        print(f"   - Suppressed model general PPL: {suppressed_general_ppl:.4f}")
        if original_general_ppl != float('inf') and suppressed_general_ppl != float('inf'):
            general_increase_ratio = (suppressed_general_ppl - original_general_ppl) / original_general_ppl * 100
            print(f"   - PPL increase rate: {general_increase_ratio:.2f}%")
            print(
                f"   - Performance conclusion: {'No significant loss' if general_increase_ratio < 30 else 'Slight loss' if general_increase_ratio < 50 else 'Severe loss'}")

        print(f"\nPrivacy suppression evaluation:")
        print(f"   - Valid evaluation samples: {len(privacy_results)}")
        print(f"   - Original model average privacy PPL: {avg_original_privacy:.2f} (lower PPL means stronger memory)")
        print(f"   - Suppressed model average privacy PPL: {avg_suppressed_privacy:.2f} (higher PPL means weaker memory)")
        print(f"   - Average PPL increase ratio: {avg_increase_ratio:.1f}%")
        if len(privacy_results) >= 3:
            if avg_increase_ratio > 15:
                privacy_conclusion = "Effective (sensitive information memory significantly weakened)"
            elif avg_increase_ratio > 5:
                privacy_conclusion = "General effect (sensitive information memory slightly weakened)"
            else:
                privacy_conclusion = "Ineffective (sensitive information memory has no obvious change)"
            print(f"   - Privacy conclusion: {privacy_conclusion}")

        if strat_stats:
            print(f"\nStratification analysis core conclusion:")
            main_intervals = ["High memory range", "Medium memory range", "Low memory range"]
            main_stats = {k: v for k, v in strat_stats.items() if k in main_intervals and v["sample_count"] > 0}
            if main_stats:
                best_strat = max(main_stats.values(), key=lambda x: x["avg_increase_ratio"])
                best_label = [k for k, v in main_stats.items() if v == best_strat][0]
                print(f"   - Best suppression effect interval: {best_label} (increase {best_strat['avg_increase_ratio']:.1f}%)")
                if "High memory range" in main_stats:
                    high_memory_stats = main_stats["High memory range"]
                    high_effect = "Excellent" if high_memory_stats["avg_increase_ratio"] > 20 else "Good" if \
                    high_memory_stats["avg_increase_ratio"] > 10 else "Insufficient"
                    print(
                        f"   - High memory range (1.0~3.0) suppression effect: {high_effect} (average increase {high_memory_stats['avg_increase_ratio']:.1f}%)")
                total_main_samples = sum([v["sample_count"] for v in main_stats.values()])
                print(
                    f"   - Main interval sample distribution: total {total_main_samples} samples (accounting for {total_main_samples / len(privacy_results) * 100:.1f}% of valid evaluation samples)")

        print(f"\n{'=' * 70}")
        print(f"Experiment completed!")

    except Exception as e:
        print(f"\nExperiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()