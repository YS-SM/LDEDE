import os
import re
import math
import time
import numpy as np
import torch
from typing import Dict, List, Tuple
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import glob
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoConfig,
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


class GPT2NamePrivacyEvaluator:
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

        self.baseline_mrr = []
        self.suppressed_mrr = []
        self.sample_details = []
        self.original_ppl = None
        self.suppressed_ppl = None

        self.max_seq_length = 512
        self.temperature = 1.0

    def _parse_gpt2_sample(self, sample_line: str) -> Dict:
        pattern = r'context: (.*?) \| privacy: (.*?) \| MRR: ([\d\.]+)'
        match = re.fullmatch(pattern, sample_line.strip())
        if not match:
            return None
        return {
            "context": match.group(1).strip(),
            "target_name": match.group(2).strip(),
            "baseline_mrr": float(match.group(3))
        }

    def load_gpt2_samples(self, sample_file_path: str, max_samples: int = 300) -> None:
        if not os.path.exists(sample_file_path):
            raise FileNotFoundError(f"Sample file does not exist: {sample_file_path}")

        self.sample_details = []
        with open(sample_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"Loaded sample file: total {len(lines)} lines")

            for line_idx, line in enumerate(lines, 1):
                if len(self.sample_details) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue

                sample = self._parse_gpt2_sample(line)
                if sample:
                    self.sample_details.append(sample)
                else:
                    print(f"[Filtered] Line {line_idx} format mismatch: {line[:50]}...")

        self.baseline_mrr = [s["baseline_mrr"] for s in self.sample_details] if self.sample_details else []
        print(f"\nSample loading completed: total {len(self.sample_details)} valid samples")

        if self.baseline_mrr:
            avg_baseline = np.mean(self.baseline_mrr)
            print(
                f"MRR statistics before suppression: Average={avg_baseline:.4f} | Max={np.max(self.baseline_mrr):.4f} | Min={np.min(self.baseline_mrr):.4f}")
        else:
            raise ValueError("No valid samples loaded, please check format")

    def _compute_single_name_mrr(self, context: str, target_name: str) -> float:
        target_tokens = self.tokenizer.tokenize(target_name)
        if not target_tokens:
            return 0.0

        context_ids = self.tokenizer.encode(
            context,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_length - len(target_tokens) - 2
        )
        context_ids = torch.tensor([context_ids]).to(self.device)

        total_mrr = 0.0
        valid_token_count = 0
        current_input_ids = context_ids.clone()

        for target_token in target_tokens:
            if current_input_ids.size(1) >= self.max_seq_length - 1:
                break

            with torch.no_grad():
                outputs = self.model(current_input_ids)
                logits = outputs.logits[:, -1, :]
                logits = logits / self.temperature

            target_token_id = self.tokenizer.convert_tokens_to_ids(target_token)
            if target_token_id == self.tokenizer.unk_token_id:
                continue

            sorted_indices = torch.argsort(logits, descending=True).squeeze()
            try:
                rank = (sorted_indices == target_token_id).nonzero().item() + 1
                rank = min(rank, 1000)
                total_mrr += 1.0 / rank
                valid_token_count += 1

                predicted_token_id = torch.tensor([[target_token_id]]).to(self.device)
                current_input_ids = torch.cat([current_input_ids, predicted_token_id], dim=1)
            except:
                total_mrr += 1.0 / 1000

        return total_mrr / valid_token_count if valid_token_count > 0 else 0.0

    def evaluate_suppressed_mrr(self) -> None:
        if not self.sample_details:
            raise ValueError("Please load samples first")

        self.suppressed_mrr = []
        for sample in tqdm(self.sample_details, desc="Calculating MRR after suppression", unit="sample"):
            try:
                mrr = self._compute_single_name_mrr(sample["context"], sample["target_name"])
                self.suppressed_mrr.append(mrr)
            except Exception as e:
                print(f"[Warning] Failed to calculate sample MRR: {str(e)[:50]}")
                self.suppressed_mrr.append(0.0)

        avg_baseline = np.mean(self.baseline_mrr)
        avg_suppressed = np.mean(self.suppressed_mrr)
        reduction_rate = (avg_baseline - avg_suppressed) / avg_baseline * 100
        print(f"\n{'=' * 50}")
        print(
            f"MRR statistics after suppression: Average={avg_suppressed:.4f} | Max={np.max(self.suppressed_mrr):.4f} | Min={np.min(self.suppressed_mrr):.4f}")
        print(f"Average MRR reduction rate: {reduction_rate:.2f}% (higher is better)")
        print(f"{'=' * 50}")

    def analyze_mrr_by_interval(self) -> None:
        if len(self.baseline_mrr) != len(self.suppressed_mrr):
            raise ValueError("MRR list lengths before/after suppression do not match")

        intervals = [
            ("High memory interval", 0.8, 0.9),
            ("Medium memory interval", 0.6, 0.8),
            ("Low memory interval", 0.4, 0.6)
        ]

        print(f"\n=== MRR stratified analysis by specified intervals ===")
        print(f"Interval type       MRR range    Sample count    Avg MRR before suppression   Avg MRR after suppression   Reduction rate    ")
        print("--------------------------------------------------------------------------")

        total_samples_in_intervals = 0
        for interval_name, min_mrr, max_mrr in intervals:
            interval_indices = [
                i for i, b_mrr in enumerate(self.baseline_mrr)
                if min_mrr <= b_mrr < max_mrr
            ]
            total_samples_in_intervals += len(interval_indices)

            if not interval_indices:
                print(
                    f"{interval_name:<12} {min_mrr:.1f}~{max_mrr:.1f}    0         No data          No data          No data   ")
                continue

            interval_baseline = [self.baseline_mrr[i] for i in interval_indices]
            interval_suppressed = [self.suppressed_mrr[i] for i in interval_indices]

            avg_b = np.mean(interval_baseline)
            avg_s = np.mean(interval_suppressed)
            reduction = (avg_b - avg_s) / avg_b * 100

            print(
                f"{interval_name:<12} {min_mrr:.1f}~{max_mrr:.1f}    {len(interval_indices):<8} {avg_b:<15.4f} {avg_s:<15.4f} {reduction:<10.2f}%")

        print(f"\nInterval analysis statistics:")
        print(f"- Total samples in three intervals: {total_samples_in_intervals}")
        print(f"- Total valid samples: {len(self.baseline_mrr)}")
        print(f"- Interval coverage rate: {total_samples_in_intervals / len(self.baseline_mrr) * 100:.2f}%")

        max_samples_interval = max(intervals, key=lambda x: sum(1 for b in self.baseline_mrr if x[1] <= b < x[2]))
        max_samples_count = sum(1 for b in self.baseline_mrr if max_samples_interval[1] <= b < max_samples_interval[2])
        print(
            f"- Interval with most samples: {max_samples_interval[0]} ({max_samples_interval[1]:.1f}~{max_samples_interval[2]:.1f}), total {max_samples_count} samples")

    def calculate_ppl_comparison(self, model_path: str) -> None:
        print(f"\n=== GPT2 Perplexity (PPL) comparison ===")
        print("Calculating suppressed model PPL...")
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
            print("Warning: Partial PPL calculation failed (possible validation set data anomaly)")
        print(f"{'=' * 50}")


def main():
    config = {
        "sample_file_path": r"D:\expt\lrp_spin\main\name\gpt2_memorized_privacy.txt",
        "neuron_path": r"D:\expt\lrp_spin\main\name\gpt2_name_neurons_voted_top5000.npy",
        "model_path": r"D:\expt\lrp_spin\WT_model\2025_11_17_gpt2",
        "valid_data_path": r"D:\expt\lrp_spin\data_process\enron_arrow\valid\final",
        "max_samples": 5000,
        "batch_size": 32,
        "use_noise": True,
        "noise_strength": 0
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
    print(f"Model loaded successfully: Number of layers={model.config.n_layer} | Sequence length={model.config.n_ctx}")

    print(f"\nLoading privacy neurons: {config['neuron_path']}")
    privacy_neurons = load_gpt2_privacy_neurons(config['neuron_path'])
    hooks = register_gpt2_privacy_hooks(
        model=model,
        privacy_neurons=privacy_neurons,
        use_noise=config['use_noise'],
        noise_strength=config['noise_strength']
    )
    print(f"\nRegistered {len(hooks)} suppression hooks")

    evaluator = GPT2NamePrivacyEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        valid_data_path=config['valid_data_path']
    )
    evaluator.load_gpt2_samples(
        sample_file_path=config['sample_file_path'],
        max_samples=config['max_samples']
    )

    evaluator.evaluate_suppressed_mrr()
    evaluator.analyze_mrr_by_interval()
    evaluator.calculate_ppl_comparison(model_path=config['model_path'])

    for hook in hooks:
        hook.remove()
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()