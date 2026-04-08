import torch
import numpy as np
import os
import re
from typing import List, Dict, Tuple
from captum.attr import LRP
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm


class GPT2FFNInputModel(torch.nn.Module):
    def __init__(self, base_gpt2: GPT2LMHeadModel):
        super().__init__()
        self.ffn_layers = [base_gpt2.transformer.h[i].mlp.c_fc for i in range(12)]
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, ffn_input: torch.Tensor, target_layer: int, target_token_positions: torch.Tensor) -> torch.Tensor:
        batch_size = ffn_input.shape[0]
        ffn_layer = self.ffn_layers[target_layer]
        ffn_output = ffn_layer(ffn_input)
        outputs = []
        for i in range(batch_size):
            target_pos = target_token_positions[i].item()
            if 0 <= target_pos < ffn_output.shape[1]:
                target_output = ffn_output[i, target_pos, :].mean()
                outputs.append(target_output.unsqueeze(0))
            else:
                outputs.append(torch.tensor(0.0, device=ffn_input.device).unsqueeze(0))
        return torch.cat(outputs, dim=0)


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
        target_text = sample["target_text"]
        target_token_len = len(tokenizer.tokenize(target_text))
        target_pos = valid_ctx_len + target_token_len - 1
        target_pos = min(target_pos, max_seq_len - 1)
        target_token_positions.append(target_pos)
    target_token_positions = torch.tensor(target_token_positions, device=device)
    return ffn_input, target_token_positions


def parse_email_samples(
        file_path: str,
        tokenizer: GPT2Tokenizer,
        max_samples: int = 2000,
        min_ppl: float = 5.0,
        max_seq_len: int = 512
) -> List[Dict]:
    samples = []
    pattern = r'【PPL：([\d\.]+)】\s*【prompt】：(.*?)\s*【target_text】：(.*?)\s*【generation_preview】：.*?'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        sample_blocks = re.split(r'={100}|\-{80}', content)
        print(f"[Initialization] Read {len(sample_blocks)} email sample blocks")

        for i, block in enumerate(sample_blocks[:2]):
            if block.strip():
                print(f"\n[Debug] Content of the {i + 1}th sample block (first 500 chars):")
                print(block.strip()[:500] + "...")

        for block in sample_blocks:
            if len(samples) >= max_samples:
                break
            block = block.strip()
            if not block:
                continue

            match = re.search(pattern, block, re.DOTALL | re.MULTILINE)
            if not match:
                print(f"[Warning] Sample format not matched, skip block (first 200 chars): {block[:200]}...")
                continue

            ppl = float(match.group(1).strip())
            prompt = match.group(2).strip()
            target_text = match.group(3).strip()

            if ppl > min_ppl:
                continue

            if not target_text or len(target_text.split()) < 5:
                print(f"[Warning] Invalid target text, skip: {target_text[:20]}")
                continue

            tokenized = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_seq_len
            )
            input_ids = tokenized["input_ids"][0].numpy()
            attention_mask = tokenized["attention_mask"][0].numpy()
            valid_ctx_len = np.sum(attention_mask)
            target_token_len = len(tokenizer.tokenize(target_text))
            target_token_position = valid_ctx_len + target_token_len - 1

            if target_token_position >= max_seq_len:
                print(
                    f"[Warning] Sample target position exceeds max sequence length ({target_token_position}/{max_seq_len}), skip: {target_text[:20]}")
                continue

            samples.append({
                "prompt": prompt,
                "target_text": target_text,
                "ppl": ppl,
                "input_ids": input_ids.astype(np.int64),
                "attention_mask": attention_mask.astype(np.int64),
                "target_token_position": target_token_position
            })

    print(f"\n[Parsing Completed] Obtained {len(samples)} valid samples (PPL≤{min_ppl}, sequence length≤{max_seq_len})")
    if samples:
        ppl_list = [s["ppl"] for s in samples]
        print(f"Sample PPL distribution: Min={min(ppl_list):.4f} | Max={max(ppl_list):.4f} | Avg={np.mean(ppl_list):.4f}")
    return samples


def compute_lrp_batch_gpt2(
        batch_samples: List[Dict],
        ffn_model: GPT2FFNInputModel,
        base_gpt2: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        device: torch.device,
        max_seq_len: int = 512
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
        ffn_input = ffn_input.detach().requires_grad_(True)
        ffn_input = ffn_input + torch.randn_like(ffn_input) * 1e-8

        lrp = LRP(model=ffn_model)
        attrs = lrp.attribute(
            inputs=(ffn_input,),
            additional_forward_args=(target_layer, target_token_positions),
            return_convergence_delta=False
        )

        attr_tensor = attrs[0] if isinstance(attrs, tuple) else attrs
        attr_tensor = torch.nan_to_num(attr_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        for i in range(batch_size):
            sample = batch_samples[i]
            target_pos = target_token_positions[i].item()
            if 0 <= target_pos < attr_tensor.shape[1]:
                layer_attr_scores = attr_tensor[i, target_pos, :].detach().cpu().numpy()
                neuron_scores = np.tile(layer_attr_scores, 4)
            else:
                neuron_scores = np.zeros(3072)
            batch_results.append({
                "target_text": sample["target_text"],
                "ppl": sample["ppl"],
                "prompt": sample["prompt"],
                "target_token_position": sample["target_token_position"],
                "privacy_neuron_scores": {target_layer: neuron_scores}
            })

    merged_results = {}
    for res in batch_results:
        key = (res["target_text"], res["prompt"])
        if key not in merged_results:
            merged_results[key] = res
        else:
            merged_results[key]["privacy_neuron_scores"].update(res["privacy_neuron_scores"])
    return list(merged_results.values())


def main():
    config = {
        "email_sample_path": r"D:\expt\lrp_spin\main\email\gpt2_high_memory_emails_generation.txt",
        "gpt2_path": r"D:\expt\lrp_spin\WT_model\2025_11_17_gpt2",
        "save_npy_path": r"D:\expt\lrp_spin\main\email\gpt2_email_lrp_results_genertarion.npy",
        "max_samples": 10000,
        "min_ppl": 10.0,
        "max_seq_len": 512,
        "batch_size": 32,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device(config["device"])
    print(f"[Environment Initialization] Using device: {device}")
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("\n=== Step 1: Parse email high-memory samples ===")
    tokenizer = GPT2Tokenizer.from_pretrained(config["gpt2_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[Tip] Set pad_token to eos_token: {tokenizer.eos_token}")

    email_samples = parse_email_samples(
        file_path=config["email_sample_path"],
        tokenizer=tokenizer,
        max_samples=config["max_samples"],
        min_ppl=config["min_ppl"],
        max_seq_len=config["max_seq_len"]
    )
    if not email_samples:
        print("[Error] No valid email samples, program terminated")
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

    print("\n=== Step 3: Batch LRP calculation (keep positive/negative signs) ===")
    all_lrp_results = []
    total_samples = len(email_samples)
    with tqdm(total=total_samples, desc="LRP calculation progress") as pbar:
        for batch_start in range(0, total_samples, config["batch_size"]):
            batch_samples = email_samples[batch_start:batch_start + config["batch_size"]]
            batch_lrp_res = compute_lrp_batch_gpt2(
                batch_samples=batch_samples,
                ffn_model=ffn_model,
                base_gpt2=base_gpt2,
                tokenizer=tokenizer,
                device=device,
                max_seq_len=config["max_seq_len"]
            )
            all_lrp_results.extend(batch_lrp_res)
            pbar.update(len(batch_samples))

    if not all_lrp_results:
        print("[Error] No valid LRP results, program terminated")
        return

    save_dir = os.path.dirname(config["save_npy_path"])
    os.makedirs(save_dir, exist_ok=True)
    np.save(config["save_npy_path"], all_lrp_results)
    print(f"\n[Success] LRP results saved (keep positive/negative signs):")
    print(f"Path: {config['save_npy_path']}")
    print(f"Valid results count: {len(all_lrp_results)}")

    print("\n=== LRP positive/negative sign distribution statistics (by layer) ===")
    layer_sign_stats = {layer: {"positive": 0, "negative": 0, "zero": 0} for layer in range(12)}
    for sample in all_lrp_results:
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
            print(f"Layer {layer_idx}: No valid attribution scores")
            continue
        pos_rate = stats["positive"] / total * 100
        neg_rate = stats["negative"] / total * 100
        zero_rate = stats["zero"] / total * 100
        print(f"Layer {layer_idx}: Positive contribution neurons={pos_rate:.1f}% | Negative contribution neurons={neg_rate:.1f}% | Zero contribution={zero_rate:.1f}%")


if __name__ == "__main__":
    main()