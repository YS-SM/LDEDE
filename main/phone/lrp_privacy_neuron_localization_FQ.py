import numpy as np
from typing import List, Dict, Tuple, Union
import json
import os
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import threading
from collections import defaultdict

progress_queue = None


def init_worker(queue):
    global progress_queue
    progress_queue = queue


def process_sample(
        sample,
        global_top_ratio: float = 0.1,
        num_neurons_per_layer: int = 3072,
        max_layer: int = 11,
        lrp_positive_threshold: float = 1e-6
):
    layer_lrps = sample["privacy_neuron_scores"]

    global_neuron_info = []
    for layer_idx in layer_lrps:
        if layer_idx > max_layer:
            continue
        lrps = np.array(layer_lrps[layer_idx], dtype=np.float32)

        positive_mask = lrps > lrp_positive_threshold
        positive_lrps = lrps[positive_mask]
        if len(positive_lrps) == 0:
            continue

        scores = positive_lrps

        neuron_indices_in_layer = np.where(positive_mask)[0]
        global_indices = layer_idx * num_neurons_per_layer + neuron_indices_in_layer

        global_neuron_info.extend(zip(scores, global_indices, positive_lrps))

    total_positive_neurons = len(global_neuron_info)
    if total_positive_neurons == 0:
        return []
    top_num = max(int(total_positive_neurons * global_top_ratio), 1)
    global_neuron_info.sort(reverse=True, key=lambda x: x[0])
    return [(idx, lrp) for (score, idx, lrp) in global_neuron_info[:top_num]]


def process_sample_chunk(
        samples_chunk,
        global_top_ratio: float = 0.1,
        num_neurons_per_layer: int = 3072,
        max_layer: int = 11,
        lrp_positive_threshold: float = 1e-6
):
    global progress_queue
    chunk_votes = defaultdict(int)
    chunk_lrp_sums = defaultdict(float)
    chunk_lrp_counts = defaultdict(int)

    for sample in samples_chunk:
        top_info = process_sample(
            sample,
            global_top_ratio=global_top_ratio,
            num_neurons_per_layer=num_neurons_per_layer,
            max_layer=max_layer,
            lrp_positive_threshold=lrp_positive_threshold
        )
        for idx, lrp in top_info:
            chunk_votes[idx] += 1
            chunk_lrp_sums[idx] += lrp
            chunk_lrp_counts[idx] += 1

    progress_queue.put(len(samples_chunk))
    return chunk_votes, chunk_lrp_sums, chunk_lrp_counts


def load_lrp_results(npy_path: str) -> List[Dict]:
    try:
        results = np.load(npy_path, allow_pickle=True).tolist()
        print(f"Successfully loaded LRP results, total {len(results)} samples")
        return results
    except Exception as e:
        print(f"Load failed: {str(e)}")
        return []


def select_privacy_neurons(
        lrp_results: List[Dict],
        top_z: int = 500,
        global_top_ratio: float = 0.1,
        vote_ratio_threshold: float = 0.5,
        layer_range: Tuple[int, int] = (0, 11),
        num_processes: int = 20,
        lrp_positive_threshold: float = 1e-6
) -> Tuple[List[Dict[str, Union[int, float]]], Dict[int, List[int]]]:
    if not lrp_results:
        print("No valid LRP results, return empty data")
        return [], {}

    total_samples = len(lrp_results)
    num_neurons_per_layer = 3072
    max_layer = 11

    print(f"\n=== Privacy neuron selection configuration (positive LRP only) ===")
    print(f"Total samples: {total_samples}")
    print(f"Single sample selection: Only retain positive LRP neurons (threshold>{lrp_positive_threshold}), then take global Top{global_top_ratio * 100}%")
    print(f"Voting threshold: Vote ratio ≥{vote_ratio_threshold * 100}% (i.e., ≥{total_samples * vote_ratio_threshold:.0f} votes)")
    print(f"Sorting rule: Descending by vote ratio (for ties, descending by average LRP → vote count → layer index)")
    print(f"Final selection: Top-{top_z} positive LRP neurons (privacy leakage promoting)")
    print(f"Layer range: {layer_range[0]}-{layer_range[1]} layers")
    print(f"Number of processes: {num_processes}")

    vote_threshold = total_samples * vote_ratio_threshold

    chunk_size = max(1, total_samples // num_processes)
    sample_chunks = [lrp_results[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]

    queue = mp.Queue()
    total_processed = 0
    progress_lock = threading.Lock()

    def progress_updater():
        nonlocal total_processed
        with tqdm(total=total_samples, desc="Processing progress (positive LRP neuron voting)", unit="sample") as pbar:
            while total_processed < total_samples:
                try:
                    chunk_done = queue.get(timeout=0.1)
                    with progress_lock:
                        total_processed += chunk_done
                        pbar.update(chunk_done)
                except:
                    continue

    updater_thread = threading.Thread(target=progress_updater, daemon=True)
    updater_thread.start()

    with mp.Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=(queue,)
    ) as pool:
        func = partial(
            process_sample_chunk,
            global_top_ratio=global_top_ratio,
            num_neurons_per_layer=num_neurons_per_layer,
            max_layer=max_layer,
            lrp_positive_threshold=lrp_positive_threshold
        )
        chunk_results = pool.map(func, sample_chunks)

    updater_thread.join()

    total_votes = defaultdict(int)
    total_lrp_sums = defaultdict(float)
    total_lrp_counts = defaultdict(int)

    for chunk_vote, chunk_lrp_sum, chunk_lrp_count in chunk_results:
        for idx, votes in chunk_vote.items():
            total_votes[idx] += votes
        for idx, lrp_sum in chunk_lrp_sum.items():
            total_lrp_sums[idx] += lrp_sum
        for idx, count in chunk_lrp_count.items():
            total_lrp_counts[idx] += count

    qualified_neurons = []
    for global_idx in total_votes:
        votes = total_votes[global_idx]
        if votes < vote_threshold:
            continue
        layer_idx = global_idx // num_neurons_per_layer
        neuron_idx = global_idx % num_neurons_per_layer
        if not (layer_range[0] <= layer_idx <= layer_range[1]):
            continue
        vote_freq = votes / total_samples
        avg_lrp = total_lrp_sums[global_idx] / total_lrp_counts[global_idx]
        if avg_lrp <= lrp_positive_threshold:
            continue
        qualified_neurons.append((
            layer_idx, neuron_idx, votes, vote_freq, avg_lrp
        ))

    qualified_neurons.sort(
        key=lambda x: (-x[3], -x[4], -x[2], x[0], x[1])
    )

    if len(qualified_neurons) < top_z:
        print(f"[Warning] Number of qualified positive LRP neurons ({len(qualified_neurons)}) is less than {top_z}, all will be selected")
        top_neurons = qualified_neurons
    else:
        top_neurons = qualified_neurons[:top_z]

    neuron_list = [
        {
            "layer_index": int(layer_idx),
            "neuron_index": int(neuron_idx),
            "LRP_value": float(avg_lrp)
        }
        for layer_idx, neuron_idx, _, _, avg_lrp in top_neurons
    ]

    compatibility_dict: Dict[int, List[int]] = defaultdict(list)
    for layer_idx, neuron_idx, _, _, _ in top_neurons:
        compatibility_dict[int(layer_idx)].append(int(neuron_idx))

    print(f"\n=== GPT2 privacy neuron selection results (positive LRP only) ===")
    print(f"Total selected positive LRP neurons: {len(neuron_list)}")
    print(f"Involved layers: {len(compatibility_dict)} layers (total 12 layers)")
    print(f"Total qualified positive LRP neurons (before truncation): {len(qualified_neurons)}")
    print(f"\nNeuron distribution by layer:")
    for layer_idx in sorted(compatibility_dict.keys()):
        neurons = compatibility_dict[layer_idx]
        neuron_count = len(neurons)
        layer_vote_freqs = [x[3] for x in top_neurons if x[0] == layer_idx]
        layer_avg_lrps = [x[4] for x in top_neurons if x[0] == layer_idx]
        avg_vote_freq = np.mean(layer_vote_freqs) if layer_vote_freqs else 0.0
        avg_lrp = np.mean(layer_avg_lrps) if layer_avg_lrps else 0.0
        top5_neurons = neurons[:5]
        print(
            f"  Layer {layer_idx}: {neuron_count} neurons | Average vote frequency {avg_vote_freq:.2%} | Average LRP {avg_lrp:.4f} | Top 5 indices: {top5_neurons}")

    print(f"\nTop 10 core positive LRP neurons details (privacy leakage promoting):")
    for i, (layer_idx, neuron_idx, votes, vote_freq, avg_lrp) in enumerate(top_neurons[:10], 1):
        print(
            f"  Rank {i}: Layer {layer_idx} | Neuron {neuron_idx} | Votes {votes} | Vote frequency {vote_freq:.2%} | LRP value {avg_lrp:.4f}")
    print("=" * 80 + "\n")

    return neuron_list, compatibility_dict


def save_neurons(
        neuron_list: List[Dict[str, Union[int, float]]],
        compatibility_dict: Dict[int, List[int]],
        save_path: str
):
    save_data = {
        "neuron_list": neuron_list,
        "compatibility_dict": compatibility_dict
    }

    np.save(save_path, save_data)
    print(f"Positive LRP neuron data saved to: {save_path}")

    json_save_path = save_path.replace(".npy", ".json")
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"JSON format backup saved to: {json_save_path}")


if __name__ == "__main__":
    if os.name == 'nt':
        mp.set_start_method('spawn', force=True)

    config = {
        "lrp_npy_path": r"D:\expt\lrp_spin\main\phone\gpt2_phone_lrp_proj_results.npy",
        "save_neuron_path": r"D:\expt\lrp_spin\main\phone\gpt2_phone_neurons_voted_top5000.npy",
        "top_z": 5000,
        "global_top_ratio": 0.5,
        "vote_ratio_threshold": 0.45,
        "layer_range": (0, 11),
        "num_processes": 20,
        "lrp_positive_threshold": 1e-6
    }

    lrp_results = load_lrp_results(config["lrp_npy_path"])
    if not lrp_results:
        exit()

    neuron_list, compatibility_dict = select_privacy_neurons(
        lrp_results=lrp_results,
        top_z=config["top_z"],
        global_top_ratio=config["global_top_ratio"],
        vote_ratio_threshold=config["vote_ratio_threshold"],
        layer_range=config["layer_range"],
        num_processes=config["num_processes"],
        lrp_positive_threshold=config["lrp_positive_threshold"]
    )

    save_neurons(neuron_list, compatibility_dict, config["save_neuron_path"])