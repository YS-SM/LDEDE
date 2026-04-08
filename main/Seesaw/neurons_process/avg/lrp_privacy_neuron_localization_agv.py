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
        num_neurons_per_layer: int = 3072,
        max_layer: int = 11,
        lrp_abs_threshold: float = 1e-6
):
    layer_lrps = sample["privacy_neuron_scores"]

    sample_neuron_info = []
    for layer_idx in layer_lrps:
        if layer_idx > max_layer:
            continue
        lrps = np.array(layer_lrps[layer_idx], dtype=np.float32)

        abs_mask = np.abs(lrps) > lrp_abs_threshold
        valid_lrps = lrps[abs_mask]
        if len(valid_lrps) == 0:
            continue

        neuron_indices_in_layer = np.where(abs_mask)[0]
        global_indices = layer_idx * num_neurons_per_layer + neuron_indices_in_layer

        sample_neuron_info.extend(zip(global_indices, valid_lrps))

    return sample_neuron_info


def process_sample_chunk(
        samples_chunk,
        num_neurons_per_layer: int = 3072,
        max_layer: int = 11,
        lrp_abs_threshold: float = 1e-6
):
    global progress_queue
    chunk_lrp_sums = defaultdict(float)
    chunk_lrp_counts = defaultdict(int)

    for sample in samples_chunk:
        neuron_info = process_sample(
            sample,
            num_neurons_per_layer=num_neurons_per_layer,
            max_layer=max_layer,
            lrp_abs_threshold=lrp_abs_threshold
        )
        for idx, lrp in neuron_info:
            chunk_lrp_sums[idx] += lrp
            chunk_lrp_counts[idx] += 1

    progress_queue.put(len(samples_chunk))
    return chunk_lrp_sums, chunk_lrp_counts


def load_lrp_results(npy_path: str) -> List[Dict]:
    try:
        results = np.load(npy_path, allow_pickle=True).tolist()
        print(f"Successfully loaded LRP results, total {len(results)} samples (retained original positive/negative signs)")
        return results
    except Exception as e:
        print(f"Loading failed: {str(e)}")
        return []


def select_all_neurons(
        lrp_results: List[Dict],
        layer_range: Tuple[int, int] = (0, 11),
        num_processes: int = 20,
        lrp_abs_threshold: float = 1e-6,
        sort_descending: bool = True
) -> Tuple[List[Dict[str, Union[int, float]]], Dict[int, List[int]]]:
    if not lrp_results:
        print("No valid LRP results, return empty data")
        return [], {}

    total_samples = len(lrp_results)
    num_neurons_per_layer = 3072
    max_layer = 11

    print(f"\n=== Neuron statistics configuration (retain all neurons, original average LRP) ===")
    print(f"Total samples: {total_samples}")
    print(f"Processing logic: Retain all valid LRP neurons (only exclude floating-point errors with absolute value <{lrp_abs_threshold})")
    print(f"Average calculation: Original LRP values (positive/negative) are directly summed and averaged (retain positive/negative signs)")
    print(f"Filter conditions: Only limit layer range {layer_range[0]}-{layer_range[1]} (can be disabled as needed)")
    print(f"Sorting rule: Sort by original average LRP value in {'descending' if sort_descending else 'ascending'} order")
    print(f"Output result: All neurons that meet the conditions (no quantity limit)")
    print(f"Number of processes: {num_processes}")

    chunk_size = max(1, total_samples // num_processes)
    sample_chunks = [lrp_results[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]

    queue = mp.Queue()
    total_processed = 0
    progress_lock = threading.Lock()

    def progress_updater():
        nonlocal total_processed
        with tqdm(total=total_samples, desc="Processing progress (count original average LRP of all neurons)", unit="sample") as pbar:
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
            num_neurons_per_layer=num_neurons_per_layer,
            max_layer=max_layer,
            lrp_abs_threshold=lrp_abs_threshold
        )
        chunk_results = pool.map(func, sample_chunks)

    updater_thread.join()

    total_lrp_sums = defaultdict(float)
    total_lrp_counts = defaultdict(int)

    for chunk_lrp_sum, chunk_lrp_count in chunk_results:
        for idx, lrp_sum in chunk_lrp_sum.items():
            total_lrp_sums[idx] += lrp_sum
        for idx, count in chunk_lrp_count.items():
            total_lrp_counts[idx] += count

    all_neurons = []
    for global_idx in total_lrp_counts:
        layer_idx = global_idx // num_neurons_per_layer
        neuron_idx = global_idx % num_neurons_per_layer

        if not (layer_range[0] <= layer_idx <= layer_range[1]):
            continue

        appear_count = total_lrp_counts[global_idx]
        original_avg_lrp = total_lrp_sums[global_idx] / appear_count

        all_neurons.append((
            layer_idx, neuron_idx, original_avg_lrp, appear_count, total_lrp_sums[global_idx]
        ))

    if sort_descending:
        all_neurons.sort(key=lambda x: (-x[2], -x[3], x[0], x[1]))
    else:
        all_neurons.sort(key=lambda x: (x[2], -x[3], x[0], x[1]))

    neuron_list = [
        {
            "layer_index": int(layer_idx),
            "neuron_index": int(neuron_idx),
            "lrp_value": float(original_avg_lrp)
        }
        for layer_idx, neuron_idx, original_avg_lrp, _, _ in all_neurons
    ]

    compatibility_dict: Dict[int, List[int]] = defaultdict(list)
    for layer_idx, neuron_idx, _, _, _ in all_neurons:
        compatibility_dict[int(layer_idx)].append(int(neuron_idx))

    total_neuron_count = len(neuron_list)
    positive_count = sum(1 for x in all_neurons if x[2] > 0)
    negative_count = sum(1 for x in all_neurons if x[2] < 0)
    zero_count = sum(1 for x in all_neurons if x[2] == 0)

    print(f"\n=== All neuron statistics results (original average LRP, retain positive/negative signs) ===")
    print(f"Total neurons: {total_neuron_count}")
    print(f"Positive/negative distribution: {positive_count} positive LRP neurons | {negative_count} negative LRP neurons | {zero_count} zero values")
    print(f"Involved layers: {len(compatibility_dict)} layers (total 12 layers)")
    print(f"Original average LRP range: {min([x[2] for x in all_neurons]):.4f} ~ {max([x[2] for x in all_neurons]):.4f}")
    print(f"\nNeuron distribution per layer:")
    for layer_idx in sorted(compatibility_dict.keys()):
        layer_neurons = [x for x in all_neurons if x[0] == layer_idx]
        neuron_count = len(layer_neurons)
        layer_avg_lrp = np.mean([x[2] for x in layer_neurons]) if layer_neurons else 0.0
        layer_positive = sum(1 for x in layer_neurons if x[2] > 0)
        layer_negative = sum(1 for x in layer_neurons if x[2] < 0)
        print(
            f"  Layer {layer_idx}: {neuron_count} neurons (positive {layer_positive} | negative {layer_negative}) | average LRP {layer_avg_lrp:.4f}")

    print(f"\nTop10 neurons (sorted by original average LRP in {'descending' if sort_descending else 'ascending'} order):")
    for i, (layer_idx, neuron_idx, original_avg_lrp, appear_count, total_lrp) in enumerate(all_neurons[:10], 1):
        contribution = "promote" if original_avg_lrp > 0 else "inhibit" if original_avg_lrp < 0 else "no contribution"
        print(
            f"  Rank {i}: Layer {layer_idx} | Neuron {neuron_idx} | Average LRP {original_avg_lrp:.4f} ({contribution}) | Appearance count {appear_count}")

    if len(all_neurons) >= 10:
        print(f"\nBottom10 neurons (sorted by original average LRP in {'descending' if sort_descending else 'ascending'} order):")
        for i, (layer_idx, neuron_idx, original_avg_lrp, appear_count, total_lrp) in enumerate(all_neurons[-10:], 1):
            contribution = "promote" if original_avg_lrp > 0 else "inhibit" if original_avg_lrp < 0 else "no contribution"
            print(
                f"  Rank {total_neuron_count - 10 + i}: Layer {layer_idx} | Neuron {neuron_idx} | Average LRP {original_avg_lrp:.4f} ({contribution}) | Appearance count {appear_count}")

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
    print(f"All neuron data saved to: {save_path}")

    json_save_path = save_path.replace(".npy", ".json")
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"JSON format backup saved to: {json_save_path}")


if __name__ == "__main__":
    if os.name == 'nt':
        mp.set_start_method('spawn', force=True)

    config = {
        "lrp_npy_path": r"D:\expt\lrp_spin\main\phone\gpt2_phone_lrp_proj_results.npy",
        "save_neuron_path": r"D:\expt\lrp_spin\main\Seesaw\gpt2_all_neurons_original_avg_phone.npy",
        "layer_range": (0, 11),
        "num_processes": 20,
        "lrp_abs_threshold": 1e-6,
        "sort_descending": True
    }

    lrp_results = load_lrp_results(config["lrp_npy_path"])
    if not lrp_results:
        exit()

    neuron_list, compatibility_dict = select_all_neurons(
        lrp_results=lrp_results,
        layer_range=config["layer_range"],
        num_processes=config["num_processes"],
        lrp_abs_threshold=config["lrp_abs_threshold"],
        sort_descending=config["sort_descending"]
    )

    save_neurons(neuron_list, compatibility_dict, config["save_neuron_path"])