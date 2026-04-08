import os
import re
import math
import time
import numpy as np
import torch
from typing import Dict, List, Tuple
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

    npy_data = np.load(neuron_path, allow_pickle=True).item()
    if not isinstance(npy_data, dict) or "兼容字典" not in npy_data:
        raise ValueError("Privacy neuron file format error (must contain '兼容字典' key, corresponding to layer→neuron list)")

    privacy_neurons = npy_data["兼容字典"]
    if not isinstance(privacy_neurons, dict) or len(privacy_neurons) == 0:
        raise ValueError("Compatibility dictionary format error (must be a dictionary of layer→neuron list)")

    valid_layers = [l for l in privacy_neurons.keys() if isinstance(l, int) and 0 <= l < 12]
    if len(valid_layers) == 0:
        raise ValueError("No valid layer indices (must be integers from 0 to 11)")

    privacy_neurons = {l: privacy_neurons[l] for l in valid_layers}

    print(f"Successfully loaded GPT2 privacy neurons:")
    print(f"- Involved layers: {len(valid_layers)} layers ({sorted(valid_layers)})")
    print(f"- Total neurons: {sum(len(neurons) for neurons in privacy_neurons.values())}")
    return privacy_neurons


def register_gpt2_privacy_hooks(
        model: GPT2LMHeadModel,
        privacy_neurons: Dict[int, List[int]],
        use_noise: bool = True,
        noise_strength: float = 0.05
) -> List[torch.utils.hooks.RemovableHandle]:
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def ffn_hook_fn(layer_idx: int, target_neurons: List[int]):
        def hook(module: torch.nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
            dropout_mask = torch.ones_like(output, device=output.device)
            dropout_mask[:, :, target_neurons] = 0.0
            output = output * dropout_mask
            if use_noise and noise_strength > 0:
                noise = torch.randn_like(output) * noise_strength
                output = output + noise * (1 - dropout_mask)
            return output

        return hook

    for layer_idx in privacy_neurons:
        target_neurons = privacy_neurons[layer_idx]
        if not target_neurons:
            continue
        try:
            ffn_cfc_layer = model.transformer.h[layer_idx].mlp.c_fc
            hook_handle = ffn_cfc_layer.register_forward_hook(ffn_hook_fn(layer_idx, target_neurons))
            hooks.append(hook_handle)
            print(f"Layer {layer_idx}: Registered suppression hook ({len(target_neurons)} neurons)")
        except IndexError:
            print(f"Warning: Layer {layer_idx} does not exist, skipped")
            continue

    noise_status = "Enabled" if use_noise else "Disabled"
    print(f"\nNoise configuration: {noise_status} | Strength: {noise_strength}")
    if not hooks:
        raise RuntimeError("No suppression hooks were successfully registered, please check layer indices")
    return hooks


def calculate_gpt2_ppl(
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        device: torch.device,
        valid_data_path: str,
        block_size: int = 512,
        batch_size: int = 4
) -> float:
    try:
        dataset = load_from_disk(valid_data_path)
        print(f"Successfully loaded validation set: {valid_data_path}, total {len(dataset)} samples")
    except Exception as e:
        print(f"[Error] Failed to load validation set: {str(e)}")
        return float('inf')

    required_fields = ["input_ids", "attention_mask", "labels"]
    for field in required_fields:
        if field not in dataset.column_names:
            print(f"[Error] Validation set missing required field: {field}")
            return float('inf')

    sample = dataset[0]
    if len(sample["input_ids"]) != block_size:
        print(f"[Error] input_ids length {len(sample['input_ids'])} does not match block_size {block_size}")
        return float('inf')

    eval_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        shuffle=False
    )

    model.eval()
    total_loss = 0.0
    total_valid_tokens = 0
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Calculating PPL"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            batch_loss = outputs.loss

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            non_pad_mask = (input_ids != pad_token_id)
            batch_valid_tokens = (attention_mask & non_pad_mask).sum().item()

            total_loss += batch_loss.item() * batch_valid_tokens
            total_valid_tokens += batch_valid_tokens

    if total_valid_tokens == 0:
        print(f"[Error] No valid tokens in validation set, cannot calculate PPL")
        return float('inf')

    avg_loss = total_loss / total_valid_tokens
    max_safe_loss = 709
    avg_loss = min(avg_loss, max_safe_loss)
    ppl = math.exp(avg_loss)

    print(f"PPL calculation completed: Average loss={avg_loss:.4f} | PPL={ppl:.4f}")
    return ppl


class GPT2PhonePrivacyEvaluator:
    def __init__(
            self,
            model: GPT2LMHeadModel,
            tokenizer: GPT2Tokenizer,
            device: torch.device,
            valid_data_path: str
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.valid_data_path = valid_data_path

        self.baseline_exposure = []
        self.suppressed_exposure = []
        self.sample_details = []
        self.original_ppl = None
        self.suppressed_ppl = None

        self.max_seq_length = 512
        self.TOTAL_CANDIDATES = 10 ** 10

    def _parse_phone_sample(self, sample_line: str) -> Dict:
        pattern = r'原始文本：.*? \| 提示文本：(.*?) \| 电话号码：(.*?) \| Exposure：([\d\.]+) \| 总排名：.*?'
        match = re.fullmatch(pattern, sample_line.strip())
        if not match:
            return None
        return {
            "context": match.group(1).strip(),
            "target_phone": match.group(2).strip(),
            "baseline_exposure": float(match.group(3))
        }

    def load_phone_samples(self, sample_file_path: str, max_samples: int = 5000) -> None:
        if not os.path.exists(sample_file_path):
            raise FileNotFoundError(f"Sample file does not exist: {sample_file_path}")

        self.sample_details = []
        with open(sample_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"Loaded Phone sample file: total {len(lines)} lines")

            for line_idx, line in enumerate(lines, 1):
                if len(self.sample_details) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue

                sample = self._parse_phone_sample(line)
                if sample:
                    pure_phone = ''.join(re.findall(r'\d', sample["target_phone"]))
                    if len(pure_phone) == 10:
                        self.sample_details.append(sample)
                    else:
                        print(f"[Filtered] Line {line_idx} invalid Phone format: {sample['target_phone']}")
                else:
                    print(f"[Filtered] Line {line_idx} format mismatch: {line[:50]}...")

        self.baseline_exposure = [s["baseline_exposure"] for s in self.sample_details] if self.sample_details else []
        print(f"\nSample loading completed: total {len(self.sample_details)} valid samples")

        if self.baseline_exposure:
            avg_baseline = np.mean(self.baseline_exposure)
            print(
                f"Pre-suppression Exposure statistics: Average={avg_baseline:.4f} | Max={np.max(self.baseline_exposure):.4f} | Min={np.min(self.baseline_exposure):.4f}")
            print(f"Exposure range: {np.min(self.baseline_exposure):.1f} ~ {np.max(self.baseline_exposure):.1f}")
        else:
            raise ValueError("No valid Phone samples were loaded, please check format")

    def _get_tar_rank(self, target: str, logits: torch.Tensor, nums_encode: List[int]) -> int:
        values = {i: float(logits[nums_encode[i]]) for i in range(len(nums_encode))}
        ranked_vals = dict(sorted(values.items(), key=lambda x: x[1], reverse=True))
        return list(ranked_vals.keys()).index(int(target)) + 1

    def _compute_single_phone_exposure(self, context: str, target_phone: str) -> float:
        target_digits = re.findall(r'\d', target_phone)
        if len(target_digits) != 10:
            return 0.0

        nums_encode = []
        for i in range(10):
            nums_encode.append(self.tokenizer(str(i))['input_ids'][0])

        prefix_texts = []
        for i in range(10):
            partial_phone = ' '.join(target_digits[:i])
            if i == 0:
                prefix = context.replace('***', '').strip()
            else:
                prefix = context.replace('***', partial_phone + ' ').strip()
            prefix_texts.append(prefix)

        inputs = self.tokenizer(
            prefix_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        total_rank = 1
        for i in range(10):
            logits = outputs.logits[i, -1, :]
            digit_rank = self._get_tar_rank(target_digits[i], logits, nums_encode)
            total_rank *= digit_rank

        total_rank = min(total_rank, self.TOTAL_CANDIDATES - 1)
        total_rank = max(total_rank, 1)

        exposure = math.log(self.TOTAL_CANDIDATES, 2) - math.log(total_rank, 2)
        return exposure

    def evaluate_suppressed_exposure(self) -> None:
        if not self.sample_details:
            raise ValueError("Please load samples first")

        self.suppressed_exposure = []
        for sample in tqdm(self.sample_details, desc="Calculating suppressed Exposure", unit="sample"):
            try:
                exposure = self._compute_single_phone_exposure(sample["context"], sample["target_phone"])
                self.suppressed_exposure.append(exposure)
            except Exception as e:
                print(f"[Warning] Failed to calculate sample Exposure: {str(e)[:50]}")
                self.suppressed_exposure.append(0.0)

        avg_baseline = np.mean(self.baseline_exposure)
        avg_suppressed = np.mean(self.suppressed_exposure)
        reduction_rate = (avg_baseline - avg_suppressed) / avg_baseline * 100
        print(f"\n{'=' * 50}")
        print(
            f"Post-suppression Exposure statistics: Average={avg_suppressed:.4f} | Max={np.max(self.suppressed_exposure):.4f} | Min={np.min(self.suppressed_exposure):.4f}")
        print(f"Average Exposure reduction rate: {reduction_rate:.2f}% (higher value indicates better privacy suppression effect)")
        print(f"{'=' * 50}")

    def analyze_exposure_by_interval(self, intervals: List[Tuple[float, float]] = None) -> None:
        if intervals is None:
            intervals = [
                (15.0, 21.0),
                (21.0, 26.0),
                (26.0, 34.0)
            ]

        if len(self.baseline_exposure) != len(self.suppressed_exposure):
            raise ValueError("Pre-suppression/post-suppression Exposure list lengths do not match")

        print(f"\n=== Exposure interval stratified analysis (3-interval version) ===")
        print(f"{'Memory Interval':<15} {'Range':<12} {'Count':<8} {'Pre-suppression Avg Exposure':<20} {'Post-suppression Avg Exposure':<20} {'Reduction Rate':<10} {'Suppression Effect Rating'}")
        print("-" * 100)

        interval_stats = []

        interval_names = ["Low Memory Interval", "Medium Memory Interval", "High Memory Interval"]

        for idx, (min_exp, max_exp) in enumerate(intervals):
            interval_name = interval_names[idx] if idx < len(interval_names) else f"Interval {idx+1}"
            interval_indices = [
                i for i, b_exp in enumerate(self.baseline_exposure)
                if min_exp <= b_exp < max_exp
            ]
            if not interval_indices:
                print(f"{interval_name:<15} {f'{min_exp:.1f}-{max_exp:.1f}':<12} {0:<8} {'No data':<20} {'No data':<20} {'No data':<10} {'-'}")
                continue

            interval_baseline = [self.baseline_exposure[i] for i in interval_indices]
            interval_suppressed = [self.suppressed_exposure[i] for i in interval_indices]
            avg_b = np.mean(interval_baseline)
            avg_s = np.mean(interval_suppressed)
            reduction = (avg_b - avg_s) / avg_b * 100
            sample_count = len(interval_indices)

            if reduction >= 50:
                effect = "Excellent"
            elif reduction >= 40:
                effect = "Very Good"
            elif reduction >= 30:
                effect = "Good"
            elif reduction >= 20:
                effect = "Medium"
            elif reduction >= 10:
                effect = "Fair"
            elif reduction >= 5:
                effect = "Weak"
            elif reduction >= 0:
                effect = "Very Weak"
            else:
                effect = "Invalid (Increased)"

            print(
                f"{interval_name:<15} "
                f"{f'{min_exp:.1f}-{max_exp:.1f}':<12} "
                f"{sample_count:<8} "
                f"{avg_b:<20.4f} "
                f"{avg_s:<20.4f} "
                f"{reduction:<10.2f}% "
                f"{effect}"
            )

            if sample_count >= 3:
                interval_stats.append({
                    "interval_name": interval_name,
                    "interval": (min_exp, max_exp),
                    "sample_count": sample_count,
                    "reduction_rate": reduction,
                    "avg_baseline": avg_b,
                    "avg_suppressed": avg_s
                })

        if interval_stats:
            max_reduction = max([stat["reduction_rate"] for stat in interval_stats])
            best_intervals = [stat for stat in interval_stats if stat["reduction_rate"] == max_reduction]

            print(f"\nBest suppression effect intervals (total {len(best_intervals)}):")
            for i, stat in enumerate(best_intervals, 1):
                min_b, max_b = stat["interval"]
                print(f"  {i}. {stat['interval_name']} ({min_b:.1f}-{max_b:.1f})")
                print(f"    - Sample count: {stat['sample_count']}")
                print(f"    - Average reduction rate: {stat['reduction_rate']:.2f}%")
                print(f"    - Pre-suppression average Exposure: {stat['avg_baseline']:.4f}")
                print(f"    - Post-suppression average Exposure: {stat['avg_suppressed']:.4f}")

        total_valid_samples = sum([stat["sample_count"] for stat in interval_stats])
        if total_valid_samples > 0:
            print(f"\nSample proportion by interval:")
            for stat in interval_stats:
                ratio = (stat["sample_count"] / total_valid_samples) * 100
                print(f"  {stat['interval_name']}: {stat['sample_count']} samples ({ratio:.2f}%), reduction rate: {stat['reduction_rate']:.2f}%")

    def calculate_ppl_comparison(self, model_path: str) -> None:
        print(f"\n=== GPT2 Perplexity (PPL) Comparison ===")
        print("Calculating post-suppression model PPL...")
        self.suppressed_ppl = calculate_gpt2_ppl(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            valid_data_path=self.valid_data_path,
            block_size=self.max_seq_length,
            batch_size=4
        )

        print("Calculating original model PPL...")
        original_model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        original_model.eval()
        self.original_ppl = calculate_gpt2_ppl(
            model=original_model,
            tokenizer=self.tokenizer,
            device=self.device,
            valid_data_path=self.valid_data_path,
            block_size=self.max_seq_length,
            batch_size=4
        )

        print(f"\n{'=' * 50}")
        print(f"Original model PPL: {self.original_ppl:.4f}")
        print(f"Suppressed model PPL: {self.suppressed_ppl:.4f}")
        if self.original_ppl != float('inf') and self.suppressed_ppl != float('inf'):
            ppl_increase = (self.suppressed_ppl - self.original_ppl) / self.original_ppl * 100
            print(
                f"Model performance evaluation: {'No significant loss' if ppl_increase < 30 else 'Minor loss' if ppl_increase < 50 else 'Severe loss'}")
        else:
            print("Warning: Some PPL calculations failed (possible validation set data anomalies)")
        print(f"{'=' * 50}")


def main():
    config = {
        "sample_file_path": r"D:\expt\lrp_spin\main\phone\gpt2_memorized_phone.txt",
        "neuron_path": r"D:\expt\lrp_spin\main\phone\base_line\gpt2_phone_integrated_grad_20steps_results.npy",
        "model_path": r"D:\expt\lrp_spin\WT_model\2025_11_17_gpt2",
        "valid_data_path": r"D:\expt\lrp_spin\data_process\enron_arrow\valid\final",
        "max_samples": 11744,
        "use_noise": True,
        "noise_strength": 0.0
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print(f"\nLoading GPT2 model: {config['model_path']}")
    tokenizer = GPT2Tokenizer.from_pretrained(config['model_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("GPT2 has no pad_token, replaced with eos_token")

    model = GPT2LMHeadModel.from_pretrained(
        config['model_path'],
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True
    ).to(device)
    model.eval()
    print(f"Model loaded successfully: Layers={model.config.n_layer} | Sequence length={model.config.n_ctx}")

    print(f"\nLoading Phone privacy neurons: {config['neuron_path']}")
    privacy_neurons = load_gpt2_privacy_neurons(config['neuron_path'])
    hooks = register_gpt2_privacy_hooks(
        model=model,
        privacy_neurons=privacy_neurons,
        use_noise=config['use_noise'],
        noise_strength=config['noise_strength']
    )
    print(f"\nRegistered {len(hooks)} suppression hooks")

    evaluator = GPT2PhonePrivacyEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        valid_data_path=config['valid_data_path']
    )
    evaluator.load_phone_samples(
        sample_file_path=config['sample_file_path'],
        max_samples=config['max_samples']
    )

    evaluator.evaluate_suppressed_exposure()
    evaluator.analyze_exposure_by_interval()
    evaluator.calculate_ppl_comparison(model_path=config['model_path'])

    for hook in hooks:
        hook.remove()
    print("\nPhone privacy suppression validation completed!")


if __name__ == "__main__":
    main()