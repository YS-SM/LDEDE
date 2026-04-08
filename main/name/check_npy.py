import numpy as np
from typing import List, Dict

def load_lrp_results(npy_path: str) -> List[Dict]:
    try:
        results = np.load(npy_path, allow_pickle=True).tolist()
        print(f"Successfully loaded LRP result file: {npy_path}")
        print(f"Total valid samples: {len(results)}")
        return results
    except Exception as e:
        print(f"Load failed: {str(e)}")
        return []

def print_lrp_overview(results: List[Dict]):
    if not results:
        print("No valid results to display")
        return

    mrr_list = [sample["mrr"] for sample in results]
    print("\n" + "="*50)
    print("LRP results overview")
    print("="*50)
    print(f"Total samples: {len(results)}")
    print(f"MRR average: {np.mean(mrr_list):.4f}")
    print(f"MRR maximum: {np.max(mrr_list):.4f}")
    print(f"MRR minimum: {np.min(mrr_list):.4f}")
    print(f"MRR standard deviation: {np.std(mrr_list):.4f}")

    sample = results[0]
    privacy_neuron_scores = sample.get("privacy_neuron_scores", {})
    layer_count = len(privacy_neuron_scores)
    if layer_count > 0:
        first_layer = next(iter(privacy_neuron_scores.keys()))
        neuron_dim = privacy_neuron_scores[first_layer].shape[0]
        print(f"\nNeuron configuration:")
        print(f"Transformer layers: {layer_count} layers (0~11)")
        print(f"Neuron dimension per layer: {neuron_dim} dimensions (c_proj input dimension)")
    else:
        print(f"\nWarning: No neuron score data for this sample")

    target_positions = [sample["target_token_position"] for sample in results]
    print(f"\nTarget position (last token generation position of privacy info):")
    print(f"Average position: {np.mean(target_positions):.1f}")
    print(f"Position range: {np.min(target_positions)} ~ {np.max(target_positions)}")
    print("="*50 + "\n")

def print_sample_detail(results: List[Dict], sample_idx: int = 0):
    if not results or sample_idx < 0 or sample_idx >= len(results):
        print(f"Invalid sample index (optional range: 0~{len(results)-1})")
        return

    sample = results[sample_idx]
    print(f"Sample {sample_idx + 1}/{len(results)} detailed information")
    print("-"*50)
    print(f"Name (privacy): {sample.get('target_name', 'None')}")
    print(f"MRR (memory intensity): {sample.get('mrr', 0.0):.4f}")
    print(f"Target position (last token generation position of privacy info): {sample.get('target_token_position', 'None')}")
    context = sample.get('context', 'None')
    print(f"Context:\n{context[:200]}..." if len(context) > 200 else f"Context:\n{context}")

    privacy_neuron_scores = sample.get("privacy_neuron_scores", {})
    if privacy_neuron_scores:
        print(f"\nFirst 3 layers neuron score statistics (3072 dimensions/layer):")
        layers = sorted(privacy_neuron_scores.keys())[:3]
        for layer in layers:
            scores = privacy_neuron_scores[layer]
            print(f"  Layer {layer}: Average={scores.mean():.6f} | Maximum={scores.max():.6f} | Minimum={scores.min():.6f} | Non-zero count={np.count_nonzero(scores):d}")
    else:
        print(f"\nWarning: No neuron score data for this sample")

    print("-"*50 + "\n")

def search_sample_by_name(results: List[Dict], name_keyword: str) -> List[Dict]:
    matched = [
        sample for sample in results
        if name_keyword.lower() in sample.get("target_name", "").lower()
    ]
    print(f"\nSearch keyword: '{name_keyword}' | Matched {len(matched)} samples")
    if matched:
        for i, sample in enumerate(matched):
            print(f"  Matched sample {i+1}: Name={sample.get('target_name', 'None')} | MRR={sample.get('mrr', 0.0):.4f} | Target position={sample.get('target_token_position', 'None')}")
    return matched

if __name__ == "__main__":
    NPY_PATH = r"D:\expt\lrp_spin\main\name\gpt2_name_lrp_proj_results.npy"

    lrp_results = load_lrp_results(NPY_PATH)
    if not lrp_results:
        exit()

    print_lrp_overview(lrp_results)

    print_sample_detail(lrp_results, sample_idx=0)

    # search_sample_by_name(lrp_results, name_keyword="Mr.")

    # if len(lrp_results) >= 10:
    #     print_sample_detail(lrp_results, sample_idx=9)