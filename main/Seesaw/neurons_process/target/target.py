import numpy as np
import json
from typing import Dict, Tuple, List
from collections import defaultdict


def load_neuron_data(npy_path: str) -> Tuple[Dict[Tuple[int, int], Dict[str, float]], Dict[str, list]]:
    try:
        data = np.load(npy_path, allow_pickle=True).item()
        if "neuron_list" not in data or "compatible_dict" not in data:
            raise ValueError("NPY file format error, missing 'neuron_list' or 'compatible_dict'")

        neuron_list = data["neuron_list"]
        neuron_map = {}

        for neuron in neuron_list:
            if "layer_index" not in neuron or "neuron_index" not in neuron:
                print(f"Warning: Skip invalid neuron (missing core fields): {neuron}")
                continue

            layer_idx = int(neuron["layer_index"])
            neuron_idx = int(neuron["neuron_index"])
            neuron_map[(layer_idx, neuron_idx)] = neuron.copy()

        print(f"Successfully loaded {npy_path}")
        print(f"  - Total valid neurons: {len(neuron_map)}")
        if neuron_map:
            sample_attrs = list(next(iter(neuron_map.values())).keys())
            print(f"  - Included attributes: {sample_attrs}")
        return neuron_map, data
    except Exception as e:
        print(f"Failed to load {npy_path}: {str(e)}")
        raise


def find_common_neurons(
        target_npy_path: str,
        phone_top5000_npy_path: str,
        output_txt_path: str,
        output_npy_path: str
):
    print("=" * 50)
    print("Start loading data...")
    target_neuron_map, _ = load_neuron_data(target_npy_path)
    phone_top5000_neuron_map, _ = load_neuron_data(phone_top5000_npy_path)
    print("=" * 50)

    print("Start finding common neurons (matching by layer_index + neuron_index)...")
    common_neurons = []
    common_compat_dict = defaultdict(list)

    for (layer, idx), target_attrs in target_neuron_map.items():
        if (layer, idx) in phone_top5000_neuron_map:
            top5000_attrs = phone_top5000_neuron_map[(layer, idx)]

            merged_attrs = {
                "layer_index": layer,
                "neuron_index": idx,
                **{k: v for k, v in target_attrs.items() if k not in ["layer_index", "neuron_index"]},
                **{f"Top5000_{k}": v for k, v in top5000_attrs.items() if k not in ["layer_index", "neuron_index"]}
            }
            common_neurons.append(merged_attrs)
            common_compat_dict[layer].append(idx)

    total_target = len(target_neuron_map)
    total_top5000 = len(phone_top5000_neuron_map)
    total_common = len(common_neurons)
    print(f"Search completed!")
    print(f"  - Total target neurons: {total_target}")
    print(f"  - Total Phone Top5000 neurons: {total_top5000}")
    print(f"  - Total common neurons: {total_common}")
    if total_target > 0:
        print(f"  - Common neurons ratio in target: {total_common / total_target * 100:.2f}%")
    if total_top5000 > 0:
        print(f"  - Common neurons ratio in Phone Top5000: {total_common / total_top5000 * 100:.2f}%")
    print("=" * 50)

    if common_neurons:
        core_headers = ["layer_index", "neuron_index"]
        other_headers = [k for k in common_neurons[0].keys() if k not in core_headers]
        headers = core_headers + sorted(other_headers)

        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(",".join(headers) + "\n")
            for neuron in common_neurons:
                values = []
                for header in headers:
                    val = neuron[header]
                    if isinstance(val, (int, float)):
                        values.append(f"{val:.6f}")
                    else:
                        values.append(str(val))
                f.write(",".join(values) + "\n")
        print(f"TXT file saved: {output_txt_path}")
    else:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("layer_index,neuron_index,target_file_related_attributes,Top5000_file_related_attributes\n")
        print(f"No common neurons found, empty TXT file generated: {output_txt_path}")

    npy_data = {
        "neuron_list": common_neurons,
        "compatible_dict": dict(common_compat_dict)
    }
    np.save(output_npy_path, npy_data)
    print(f"NPY file saved: {output_npy_path}")

    json_path = output_npy_path.replace(".npy", ".json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(npy_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON backup file saved: {json_path}")
    print("=" * 50)

    if common_compat_dict:
        print("Common neurons layer distribution:")
        for layer in sorted(common_compat_dict.keys()):
            neuron_count = len(common_compat_dict[layer])
            print(f"  - Layer {layer}: {neuron_count} neurons")
    print("=" * 50)


if __name__ == "__main__":
    config = {
        "target_npy_path": r"D:\expt\lrp_spin\main\Seesaw\neurons_process\Coupling\target_neurons_name_neg_phone_pos.npy",
        "phone_top5000_npy_path": r"D:\expt\lrp_spin\main\phone\gpt2_phone_neurons_voted_top5000.npy",
        "output_txt_path": r"D:\expt\lrp_spin\main\Seesaw\neurons_process\target\common_neurons_target_vs_phone_top5000.txt",
        "output_npy_path": r"D:\expt\lrp_spin\main\Seesaw\neurons_process\target\common_neurons_target_vs_phone_top5000.npy"
    }

    find_common_neurons(
        target_npy_path=config["target_npy_path"],
        phone_top5000_npy_path=config["phone_top5000_npy_path"],
        output_txt_path=config["output_txt_path"],
        output_npy_path=config["output_npy_path"]
    )