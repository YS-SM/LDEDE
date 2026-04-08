import numpy as np
import json
from typing import Dict, Tuple, List
from collections import defaultdict

def load_neuron_data(npy_path: str) -> Tuple[Dict[Tuple[int, int], float], Dict[str, list]]:
    try:
        data = np.load(npy_path, allow_pickle=True).item()
        neuron_list = data["neuron_list"]
        neuron_map = {}
        for neuron in neuron_list:
            layer_idx = neuron["layer_index"]
            neuron_idx = neuron["neuron_index"]
            lrp_val = neuron["lrp_value"]
            neuron_map[(layer_idx, neuron_idx)] = lrp_val
        print(f"Successfully loaded {npy_path}, total {len(neuron_map)} neurons")
        return neuron_map, data
    except Exception as e:
        print(f"Failed to load {npy_path}: {str(e)}")
        raise

def find_target_neurons(
        name_npy_path: str,
        phone_npy_path: str,
        output_txt_path: str,
        output_npy_path: str
):
    print("Start loading data...")
    name_neuron_map, _ = load_neuron_data(name_npy_path)
    phone_neuron_map, _ = load_neuron_data(phone_npy_path)

    print("Start filtering target neurons...")
    target_neurons_txt = []
    target_neurons_npy = []
    target_compatibility_dict = defaultdict(list)

    for (layer_idx, neuron_idx), name_lrp in name_neuron_map.items():
        if name_lrp < 0:
            if (layer_idx, neuron_idx) in phone_neuron_map:
                phone_lrp = phone_neuron_map[(layer_idx, neuron_idx)]
                if phone_lrp > 0:
                    target_neurons_txt.append({
                        "layer_index": layer_idx,
                        "neuron_index": neuron_idx,
                        "LRP1(name)": name_lrp,
                        "LRP2(phone)": phone_lrp
                    })

                    target_neurons_npy.append({
                        "layer_index": int(layer_idx),
                        "neuron_index": int(neuron_idx),
                        "lrp_value(name)": float(name_lrp),
                        "lrp_value(phone)": float(phone_lrp)
                    })

                    target_compatibility_dict[int(layer_idx)].append(int(neuron_idx))

    print(f"Filtering completed, found {len(target_neurons_txt)} target neurons in total")
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("layer_index,neuron_index,LRP1(name),LRP2(phone)\n")
        for neuron in target_neurons_txt:
            line = f"{neuron['layer_index']},{neuron['neuron_index']}," \
                   f"{neuron['LRP1(name)']:.6f},{neuron['LRP2(phone)']:.6f}\n"
            f.write(line)
    print(f"TXT results saved to: {output_txt_path}")

    npy_save_data = {
        "neuron_list": target_neurons_npy,
        "compatibility_dict": dict(target_compatibility_dict)
    }
    np.save(output_npy_path, npy_save_data)
    print(f"NPY results saved to: {output_npy_path}")

    json_save_path = output_npy_path.replace(".npy", ".json")
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(npy_save_data, f, indent=2, ensure_ascii=False)
    print(f"JSON backup saved to: {json_save_path}")

    if target_neurons_txt:
        layers = set(neuron["layer_index"] for neuron in target_neurons_txt)
        print(f"\n=== Target neuron statistics ===")
        print(f"Total count: {len(target_neurons_txt)}")
        print(f"Number of layers: {len(layers)} ({sorted(layers)})")
        print(f"LRP1(name) range (negative): {min(n['LRP1(name)'] for n in target_neurons_txt):.6f} ~ 0.0")
        print(f"LRP2(phone) range (positive): 0.0 ~ {max(n['LRP2(phone)'] for n in target_neurons_txt):.6f}")
    else:
        print("\nNo neurons meeting the criteria were found")

if __name__ == "__main__":
    config = {
        "name_npy_path": r"D:\expt\lrp_spin\main\Seesaw\gpt2_all_neurons_original_avg_name.npy",
        "phone_npy_path": r"D:\expt\lrp_spin\main\Seesaw\gpt2_all_neurons_original_avg_phone.npy",
        "output_txt_path": r"D:\expt\lrp_spin\main\Seesaw\target_neurons_name_neg_phone_pos.txt",
        "output_npy_path": r"D:\expt\lrp_spin\main\Seesaw\target_neurons_name_neg_phone_pos.npy"
    }

    find_target_neurons(
        name_npy_path=config["name_npy_path"],
        phone_npy_path=config["phone_npy_path"],
        output_txt_path=config["output_txt_path"],
        output_npy_path=config["output_npy_path"]
    )