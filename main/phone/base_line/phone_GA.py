import torch
import numpy as np
import os
import re
from typing import List, Dict, Tuple
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm


class GPT2FFNInputModel(torch.nn.Module):
    def __init__(self, base_gpt2: GPT2LMHeadModel):
        super().__init__()
        self.ffn_layers = [base_gpt2.transformer.h[i].mlp.c_fc for i in range(12)]
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, ffn_input: torch.Tensor, target_layer: int) -> torch.Tensor:
        ffn_layer = self.ffn_layers[target_layer]
        ffn_output = ffn_layer(ffn_input)
        return ffn_output


def scaled_input(emb: torch.Tensor, num_steps: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    baseline = torch.zeros_like(emb)
    step = (emb - baseline) / num_steps
    scaled_embs = torch.cat([baseline + step * i for i in range(num_steps)], dim=0)
    return scaled_embs, step[0]


def precompute_ffn_input(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        base_gpt2: GPT2LMHeadModel,
        target_layer: int,
        device: torch.device,
        batch_samples: List[Dict],
        tokenizer: GPT2Tokenizer,
        max_seq_len: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    base_gpt2.eval()
    with torch.no_grad():
        wte = base_gpt2.transformer.wte(input_ids)
        wpe = base_gpt2.transformer.wpe(
            torch.arange(input_ids.shape[1], device=device).unsqueeze(0).repeat(input_ids.shape[0], 1))
        hidden_state = wte + wpe
        hidden_state = torch.clamp(hidden_state, min=-10.0, max=10.0)

        for layer_idx in range(target_layer + 1):
            layer = base_gpt2.transformer.h[layer_idx]

            attn_mask = attention_mask[:, None, None, :]
            attn_mask = (1.0 - attn_mask) * -10000.0
            attn_outputs = layer.attn(hidden_states=hidden_state, attention_mask=attn_mask)
            attn_output = attn_outputs[0]
            attn_output = torch.clamp(attn_output, min=-10.0, max=10.0)

            if layer_idx == target_layer:
                ffn_input = hidden_state
                break

            hidden_state = attn_output + hidden_state
            hidden_state = torch.clamp(hidden_state, min=-10.0, max=10.0)

    target_token_positions = []
    for i in range(input_ids.shape[0]):
        sample = batch_samples[i]
        valid_ctx_len = np.sum(attention_mask[i].cpu().numpy() == 1)
        target_phone = sample["target_name"]
        target_token_len = len(tokenizer.tokenize(target_phone))
        target_pos = valid_ctx_len + target_token_len - 1
        target_pos = min(target_pos, max_seq_len - 1)
        target_token_positions.append(target_pos)
    target_token_positions = torch.tensor(target_token_positions, device=device)
    return ffn_input, target_token_positions


def parse_phone_samples(
        file_path: str,
        tokenizer: GPT2Tokenizer,
        max_samples: int = 2000,
        min_exposure: float = 10.0,
        max_seq_len: int = 512
) -> List[Dict]:
    samples = []
    pattern = r'原始文本：.*? \| 提示文本：(.*?) \| 电话号码：(.*?) \| Exposure：([\d\.]+) \| 总排名：.*?'
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"[Initialization] Read {len(lines)} lines from Phone sample file")
        for line_num, line in enumerate(lines, 1):
            if len(samples) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            match = re.fullmatch(pattern, line)
            if not match:
                continue
            context = match.group(1).strip()
            target_phone = match.group(2).strip()
            exposure = float(match.group(3))

            if exposure < min_exposure:
                continue
            tokenized = tokenizer(
                context,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_seq_len
            )
            input_ids = tokenized["input_ids"][0].numpy()
            attention_mask = tokenized["attention_mask"][0].numpy()
            valid_context_len = np.sum(attention_mask)
            target_token_position = valid_context_len + len(tokenizer.tokenize(target_phone)) - 1
            if target_token_position >= max_seq_len:
                continue
            samples.append({
                "context": context,
                "target_name": target_phone,
                "mrr": exposure,
                "input_ids": input_ids.astype(np.int64),
                "attention_mask": attention_mask.astype(np.int64),
                "target_token_position": target_token_position
            })
    print(f"\n[Parsing completed] Obtained {len(samples)} valid samples (Exposure≥{min_exposure})")
    return samples


def compute_integrated_grad_batch_gpt2(
        batch_samples: List[Dict],
        ffn_model: GPT2FFNInputModel,
        base_gpt2: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        device: torch.device,
        max_seq_len: int = 512,
        num_steps: int = 20
) -> List[Dict]:
    if not batch_samples:
        return []
    batch_size = len(batch_samples)
    input_ids_batch = torch.tensor(
        np.array([s["input_ids"] for s in batch_samples]),
        dtype=torch.long,
        device=device
    )
    attention_mask_batch = torch.tensor(
        np.array([s["attention_mask"] for s in batch_samples]),
        dtype=torch.long,
        device=device
    )
    target_token_positions = torch.tensor(
        [s["target_token_position"] for s in batch_samples],
        dtype=torch.long,
        device=device
    )

    batch_results = []
    for target_layer in range(12):
        ffn_input, _ = precompute_ffn_input(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            base_gpt2=base_gpt2,
            target_layer=target_layer,
            device=device,
            batch_samples=batch_samples,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len
        )
        ffn_input = ffn_input.detach()

        for sample_idx in range(batch_size):
            sample = batch_samples[sample_idx]
            target_pos = target_token_positions[sample_idx].item()

            sample_ffn_input = ffn_input[sample_idx, target_pos, :].unsqueeze(0)

            if sample_ffn_input.shape[1] != 768:
                neuron_scores = np.zeros(3072)
                batch_results.append({
                    "target_name": sample["target_name"],
                    "mrr": sample["mrr"],
                    "context": sample["context"],
                    "target_token_position": sample["target_token_position"],
                    "privacy_neuron_scores": {target_layer: neuron_scores}
                })
                continue

            scaled_inputs, step = scaled_input(
                emb=sample_ffn_input,
                num_steps=num_steps
            )
            scaled_inputs = scaled_inputs.requires_grad_(True)

            integrated_grad = torch.zeros_like(step, device=device)
            for step_idx in range(num_steps):
                step_input = scaled_inputs[step_idx:step_idx+1]

                ffn_output = ffn_model(
                    ffn_input=step_input,
                    target_layer=target_layer
                )

                output_sum = ffn_output.sum()
                output_sum.backward()

                integrated_grad += scaled_inputs.grad[step_idx:step_idx+1].squeeze(0)

                scaled_inputs.grad.zero_()

            integrated_grad = (integrated_grad / num_steps) * step
            neuron_scores = np.tile(integrated_grad.detach().cpu().numpy(), 4)
            neuron_scores = np.nan_to_num(neuron_scores, nan=0.0, posinf=0.0, neginf=0.0)

            batch_results.append({
                "target_name": sample["target_name"],
                "mrr": sample["mrr"],
                "context": sample["context"],
                "target_token_position": sample["target_token_position"],
                "privacy_neuron_scores": {target_layer: neuron_scores}
            })

    merged_results = {}
    for res in batch_results:
        key = (res["target_name"], res["context"])
        if key not in merged_results:
            merged_results[key] = res
        else:
            merged_results[key]["privacy_neuron_scores"].update(res["privacy_neuron_scores"])
    return list(merged_results.values())


def main():
    config = {
        "phone_sample_path": r"D:\expt\lrp_spin\main\phone\gpt2_memorized_phone.txt",
        "gpt2_path": r"D:\expt\lrp_spin\WT_model\2025_11_17_gpt2",
        "save_npy_path": r"D:\expt\lrp_spin\main\phone\base_line\gpt2_phone_integrated_grad_20steps_results.npy",
        "max_samples": 1024,
        "min_exposure": 14.0,
        "max_seq_len": 512,
        "batch_size": 32,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_steps": 50
    }

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device(config["device"])
    print(f"[Environment Initialization] Using device: {device}")
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("\n=== Step 1: Parse high-memory Phone samples ===")
    tokenizer = GPT2Tokenizer.from_pretrained(config["gpt2_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[Notice] Set pad_token to eos_token: {tokenizer.eos_token}")
    phone_samples = parse_phone_samples(
        file_path=config["phone_sample_path"],
        tokenizer=tokenizer,
        max_samples=config["max_samples"],
        min_exposure=config["min_exposure"],
        max_seq_len=config["max_seq_len"]
    )
    if not phone_samples:
        print("[Error] No valid samples, program terminated")
        return

    print("\n=== Step 2: Load GPT2 model ===")
    base_gpt2 = GPT2LMHeadModel.from_pretrained(
        config["gpt2_path"],
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True
    ).to(device)
    base_gpt2.eval()
    print(f"Model loaded | Layers: {base_gpt2.config.n_layer} | Precision: float32 | Sequence length: {config['max_seq_len']}")

    ffn_model = GPT2FFNInputModel(base_gpt2).to(device)
    ffn_model.eval()

    print(f"\n=== Step 3: Integrated Gradients Calculation (20 steps integration, no internal batching) ===")
    print(f"Integrated Gradients config: Integration steps={config['num_steps']} | No internal batching | Keep positive/negative signs")
    all_ig_results = []
    total_samples = len(phone_samples)
    with tqdm(total=total_samples, desc="Sample batch progress") as pbar_sample:
        for batch_start in range(0, total_samples, config["batch_size"]):
            batch_samples = phone_samples[batch_start:batch_start + config["batch_size"]]
            batch_ig_res = compute_integrated_grad_batch_gpt2(
                batch_samples=batch_samples,
                ffn_model=ffn_model,
                base_gpt2=base_gpt2,
                tokenizer=tokenizer,
                device=device,
                max_seq_len=config["max_seq_len"],
                num_steps=config["num_steps"]
            )
            all_ig_results.extend(batch_ig_res)
            pbar_sample.update(len(batch_samples))

    if not all_ig_results:
        print("[Error] No valid Integrated Gradients results, program terminated")
        return

    save_dir = os.path.dirname(config["save_npy_path"])
    os.makedirs(save_dir, exist_ok=True)
    np.save(config["save_npy_path"], all_ig_results)
    print(f"\n[Success] 20-step Integrated Gradients results saved (keep positive/negative signs):")
    print(f"Path: {config['save_npy_path']}")
    print(f"Valid results count: {len(all_ig_results)}")

    print("\n=== Full layer validity verification (aligned with BERT) ===")
    layer_valid_count = {layer: 0 for layer in range(12)}
    for sample in all_ig_results:
        for layer_idx in range(12):
            if layer_idx in sample["privacy_neuron_scores"]:
                scores = sample["privacy_neuron_scores"][layer_idx]
                if not np.isnan(scores).any() and np.abs(scores).sum() > 0:
                    layer_valid_count[layer_idx] += 1
    for layer_idx in sorted(layer_valid_count.keys()):
        valid_rate = layer_valid_count[layer_idx] / len(all_ig_results) * 100
        print(f"Layer {layer_idx}: Valid sample rate={valid_rate:.1f}% ({layer_valid_count[layer_idx]}/{len(all_ig_results)})")

    print("\n=== 20-step Integrated Gradients positive/negative sign distribution statistics ===")
    layer_sign_stats = {layer: {"positive": 0, "negative": 0, "zero": 0} for layer in range(12)}
    for sample in all_ig_results:
        for layer_idx in range(12):
            if layer_idx not in sample["privacy_neuron_scores"]:
                continue
            scores = sample["privacy_neuron_scores"][layer_idx]
            positive_count = np.sum(scores > 1e-6)
            negative_count = np.sum(scores < -1e-6)
            zero_count = len(scores) - positive_count - negative_count
            layer_sign_stats[layer_idx]["positive"] += positive_count
            layer_sign_stats[layer_idx]["negative"] += negative_count
            layer_sign_stats[layer_idx]["zero"] += zero_count

    for layer_idx in sorted(layer_sign_stats.keys()):
        stats = layer_sign_stats[layer_idx]
        total = stats["positive"] + stats["negative"] + stats["zero"]
        if total == 0:
            print(f"Layer {layer_idx}: No valid scores")
            continue
        pos_rate = stats["positive"] / total * 100
        neg_rate = stats["negative"] / total * 100
        zero_rate = stats["zero"] / total * 100
        print(f"Layer {layer_idx}: Positive contribution neurons={pos_rate:.1f}% | Negative contribution neurons={neg_rate:.1f}% | Zero contribution={zero_rate:.1f}%")


if __name__ == "__main__":
    main()