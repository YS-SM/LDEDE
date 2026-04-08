import numpy as np
import json
from typing import Dict, Tuple, List
from collections import defaultdict


def load_neuron_data(npy_path: str) -> Tuple[Dict[Tuple[int, int], Dict[str, float]], Dict[str, list]]:
    data = np.load(npy_path, allow_pickle=True).item()
    if "neuron_list" not in data or "compatible_dict" not in data:
        raise ValueError("NPY file format error, missing 'neuron_list' or 'compatible_dict'")

    neuron_map = {}
    for neuron in data["neuron_list"]:
        if "layer_index" not in neuron or "neuron_index" not in neuron:
            print(f"Warning: Skip invalid neuron (missing core fields): {neuron}")
            continue
        layer_idx = int(neuron["layer_index"])
        neuron_idx = int(neuron["neuron_index"])
        neuron_map[(layer_idx, neuron_idx)] = neuron.copy()

    print(f"Successfully loaded {npy_path}")
    print(f"  - Total valid neurons: {len(neuron_map)}")
    return neuron_map, data
except Exception as e:
    print(f"Failed to load {npy_path}: {str(e)}")
    raise


def remove_common_neurons(
        phone_top5000_npy_path: str,
        common_neurons_npy_path: str,
        output_npy_path: str,
        output_txt_path: str
):
    print("=" * 50)
    print("Start loading data...")
    phone_top5000_map, phone_top5000_original = load_neuron_data(phone_top5000_npy_path)
    common_neurons_map, _ = load_neuron_data(common_neurons_npy_path)
    print("=" * 50)

    print("Start removing common neurons...")
    remaining_neurons = []
    remaining_compat_dict = defaultdict(list)

    for (layer_idx, neuron_idx), neuron_attrs in phone_top5000_map.items():
        if (layer_idx, neuron_idx) not in common_neurons_map:
            remaining_neurons.append(neuron_attrs)
            remaining_compat_dict[layer_idx].append(neuron_idx)

    total_top5000 = len(phone_top5000_map)
    total_common = len(common_neurons_map)
    total_remaining = len(remaining_neurons)

    print(f"Processing completed!")
    print(f"  - Total neurons in original phone Top5000: {total_top5000}")
    print(f"  - Number of common neurons to remove: {total_common}")
    print(f"  - Remaining neurons after removal: {total_remaining}")
    print(f"  - Removal ratio: {total_common / total_top5000 * 100:.2f}%")
    print("=" * 50)

    npy_save_data = {
        "neuron_list": remaining_neurons,
        "compatible_dict": dict(remaining_compat_dict)
    }
    np.save(output_npy_path, npy_save_data)
    print(f"NPY file saved (same format as original Top5000): {output_npy_path}")

    if remaining_neurons:
        headers = list(remaining_neurons[0].keys())
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(",".join(headers) + "\n")
            for neuron in remaining_neurons:
                values = []
                for header in headers:
                    val = neuron[header]
                    if isinstance(val, (int, float)):
                        values.append(f"{val:.6f}")
                    else:
                        values.append(str(val))
                f.write(",".join(values) + "\n")
    else:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("layer_index,neuron_index,LRP_value\n")
    print(f"TXT backup file saved: {output_txt_path}")

    if remaining_compat_dict:
        print("\nRemaining neurons layer distribution:")
        for layer in sorted(remaining_compat_dict.keys()):
            count = len(remaining_compat_dict[layer])
            print(f"  - Layer {layer}: {count} neurons")
    print("=" * 50)


if __name__ == "__main__":
    config = {
        "phone_top5000_npy_path": r"D:\expt\lrp_spin\main\phone\gpt2_phone_neurons_voted_top5000.npy",
        "common_neurons_npy_path": r"D:\expt\lrp_spin\main\Seesaw\neurons_process\target\common_neurons_target_vs_phone_top5000.npy",
        "output_npy_path": r"D:\expt\lrp_spin\main\Seesaw\neurons_process\wu_targe\gpt2_phone_neurons_voted_top5000_removed_common.npy",
        "output_txt_path": r"D:\expt\lrp_spin\main\Seesaw\neurons_process\wu_targe\gpt2_phone_neurons_voted_top5000_removed_common.txt"
    }

    remove_common_neurons(
        phone_top5000_npy_path=config["phone_top5000_npy_path"],
        common_neurons_npy_path=config["common_neurons_npy_path"],
        output_npy_path=config["output_npy_path"],
        output_txt_path=config["output_txt_path"]
    )