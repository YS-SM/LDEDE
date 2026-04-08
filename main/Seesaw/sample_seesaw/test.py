import os
import re
from typing import Dict, List, Set
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MRRIncreasedSamplesComparator:
    def __init__(self):
        self.sample_pattern = re.compile(
            r"【Sample\d+】\s+"
            r"Original line：(context: .*? \| privacy: .*? \| MRR: [\d\.]+)\s+"
            r"Suppressed MRR：([\d\.]+)\s+"
            r"MRR change：([\d\.]+)\s+"
            r"MRR increase rate：([\d\.]+%)\s+"
            r"Context：(.*?)\s+"
            r"Target name：(.*?)\s+"
            r"-{5,}",
            re.DOTALL
        )
        self.sample_key_func = lambda ctx, target: (ctx.strip(), target.strip())

    def parse_sample_file(self, file_path: str) -> Dict[tuple, Dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        samples = {}
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        matches = self.sample_pattern.findall(content)
        print(f"Parsed {len(matches)} samples from {os.path.basename(file_path)}")

        for idx, match in enumerate(matches, 1):
            original_line, suppressed_mrr, mrr_change, mrr_change_rate, context, target_name = match
            sample_key = self.sample_key_func(context, target_name)

            samples[sample_key] = {
                "sample_id": idx,
                "original_line": original_line.strip(),
                "suppressed_mrr": float(suppressed_mrr),
                "mrr_change": float(mrr_change),
                "mrr_change_rate": mrr_change_rate.strip(),
                "context": context.strip(),
                "target_name": target_name.strip(),
                "source_file": os.path.basename(file_path)
            }

        return samples

    def compare_samples(self, file1_path: str, file2_path: str, output_file: str = "mrr_samples_comparison_report.txt",
                        plot_increase_output: str = "mrr_top10_increase_histogram.png",
                        plot_decrease_output: str = "mrr_top10_decrease_histogram.png"):
        print("=" * 80)
        print("Start comparing MRR increased sample files")
        print(f"Baseline file: {file1_path}")
        print(f"Comparison file: {file2_path}")
        print("=" * 80)

        try:
            file1_samples = self.parse_sample_file(file1_path)
            file2_samples = self.parse_sample_file(file2_path)
        except Exception as e:
            print(f"Failed to parse files: {str(e)}")
            return

        file2_unique_samples = []
        for sample_key, sample_info in file2_samples.items():
            if sample_key not in file1_samples:
                file2_unique_samples.append(sample_info)

        same_sample_better_mrr = []
        same_sample_worse_mrr = []

        for sample_key, file2_sample in file2_samples.items():
            if sample_key in file1_samples:
                file1_sample = file1_samples[sample_key]
                if file2_sample["suppressed_mrr"] > file1_sample["suppressed_mrr"]:
                    mrr_diff = file2_sample["suppressed_mrr"] - file1_sample["suppressed_mrr"]
                    mrr_diff_rate = (mrr_diff / file1_sample["suppressed_mrr"]) * 100
                    same_sample_better_mrr.append({
                        "file1_sample": file1_sample,
                        "file2_sample": file2_sample,
                        "mrr_diff": mrr_diff,
                        "mrr_diff_rate": mrr_diff_rate
                    })
                elif file2_sample["suppressed_mrr"] < file1_sample["suppressed_mrr"]:
                    mrr_diff = file2_sample["suppressed_mrr"] - file1_sample["suppressed_mrr"]
                    mrr_diff_rate = (mrr_diff / file1_sample["suppressed_mrr"]) * 100
                    same_sample_worse_mrr.append({
                        "file1_sample": file1_sample,
                        "file2_sample": file2_sample,
                        "mrr_diff": mrr_diff,
                        "mrr_diff_rate": mrr_diff_rate
                    })

        same_sample_better_mrr.sort(key=lambda x: x["mrr_diff_rate"], reverse=True)
        same_sample_worse_mrr.sort(key=lambda x: x["mrr_diff_rate"])

        unique_increase_samples = []
        seen_increase_pairs = set()
        for item in same_sample_better_mrr:
            b_mrr = round(item["file1_sample"]["suppressed_mrr"], 4)
            c_mrr = round(item["file2_sample"]["suppressed_mrr"], 4)
            if (b_mrr, c_mrr) not in seen_increase_pairs:
                seen_increase_pairs.add((b_mrr, c_mrr))
                unique_increase_samples.append(item)
                if len(unique_increase_samples) >= 10:
                    break

        unique_decrease_samples = []
        seen_decrease_pairs = set()
        for item in same_sample_worse_mrr:
            b_mrr = round(item["file1_sample"]["suppressed_mrr"], 4)
            c_mrr = round(item["file2_sample"]["suppressed_mrr"], 4)
            if (b_mrr, c_mrr) not in seen_decrease_pairs:
                seen_decrease_pairs.add((b_mrr, c_mrr))
                unique_decrease_samples.append(item)
                if len(unique_decrease_samples) >= 10:
                    break

        self.generate_report(
            file1_samples=file1_samples,
            file2_samples=file2_samples,
            file2_unique_samples=file2_unique_samples,
            same_sample_better_mrr=same_sample_better_mrr,
            same_sample_worse_mrr=same_sample_worse_mrr,
            output_file=output_file,
            file1_path=file1_path,
            file2_path=file2_path
        )

        self.generate_mrr_increase_histogram(unique_increase_samples, plot_increase_output)
        self.generate_mrr_decrease_histogram(unique_decrease_samples, plot_decrease_output)

        print("\n" + "=" * 80)
        print("Comparison completed!")
        print(f"Comparison report saved to: {os.path.abspath(output_file)}")
        print(f"MRR increase sample histogram saved to: {os.path.abspath(plot_increase_output)}")
        print(f"MRR decrease sample histogram saved to: {os.path.abspath(plot_decrease_output)}")
        print(f"Statistics:")
        print(f"- Total samples in baseline file: {len(file1_samples)}")
        print(f"- Total samples in comparison file: {len(file2_samples)}")
        print(f"- Unique samples in comparison file: {len(file2_unique_samples)}")
        print(f"- Number of same samples with higher MRR in comparison file: {len(same_sample_better_mrr)}")
        print(f"- Number of same samples with lower MRR in comparison file: {len(same_sample_worse_mrr)}")
        print(f"- Generated histogram for {len(unique_increase_samples)} unique MRR increase samples")
        print(f"- Generated histogram for {len(unique_decrease_samples)} unique MRR decrease samples")
        print("=" * 80)

    def generate_report(self, file1_samples: Dict, file2_samples: Dict,
                        file2_unique_samples: List[Dict], same_sample_better_mrr: List[Dict],
                        same_sample_worse_mrr: List[Dict], output_file: str,
                        file1_path: str, file2_path: str):
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("MRR Increased Samples File Comparison Report\n")
            f.write("=" * 100 + "\n")
            f.write(f"Generation time: {self.get_current_time()}\n")
            f.write(f"Baseline file (File 1): {os.path.basename(file1_path)}\n")
            f.write(f"Comparison file (File 2): {os.path.basename(file2_path)}\n")
            f.write(f"Baseline file path: {file1_path}\n")
            f.write(f"Comparison file path: {file2_path}\n")
            f.write("\n")

            f.write("1. Overall Statistics\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total samples in baseline file: {len(file1_samples)}\n")
            f.write(f"Total samples in comparison file: {len(file2_samples)}\n")
            f.write(f"Unique samples in comparison file: {len(file2_unique_samples)}\n")
            f.write(f"Number of same samples with higher MRR in comparison file: {len(same_sample_better_mrr)}\n")
            f.write(f"Number of same samples with lower MRR in comparison file: {len(same_sample_worse_mrr)}\n")
            f.write(f"Number of common samples in both files: {len(set(file1_samples.keys()) & set(file2_samples.keys()))}\n")
            f.write("\n")

            f.write("2. Unique Samples in Comparison File (Not in File 1)\n")
            f.write("-" * 50 + "\n")
            if file2_unique_samples:
                f.write(f"Total {len(file2_unique_samples)} unique samples:\n\n")
                for idx, sample in enumerate(file2_unique_samples, 1):
                    f.write(f"【Unique Sample {idx}】\n")
                    f.write(f"Original line: {sample['original_line']}\n")
                    f.write(f"Suppressed MRR: {sample['suppressed_mrr']:.4f}\n")
                    f.write(f"MRR change: {sample['mrr_change']:.4f}\n")
                    f.write(f"MRR increase rate: {sample['mrr_change_rate']}\n")
                    f.write(f"Context: {sample['context']}\n")
                    f.write(f"Target name: {sample['target_name']}\n")
                    f.write("-" * 80 + "\n\n")
            else:
                f.write("No unique samples in comparison file (all samples exist in file 1)\n\n")

            f.write("3. Same Text Samples with Higher Suppressed MRR in Comparison File\n")
            f.write("-" * 50 + "\n")
            if same_sample_better_mrr:
                f.write(f"Total {len(same_sample_better_mrr)} samples (sorted by MRR increase rate):\n\n")
                for idx, item in enumerate(same_sample_better_mrr, 1):
                    file1_sample = item["file1_sample"]
                    file2_sample = item["file2_sample"]

                    f.write(f"【Increased Sample {idx}】\n")
                    f.write(f"Original line: {file1_sample['original_line']}\n")
                    f.write(f"Target name: {file1_sample['target_name']}\n")
                    f.write(f"File 1 suppressed MRR: {file1_sample['suppressed_mrr']:.4f}\n")
                    f.write(f"File 2 suppressed MRR: {file2_sample['suppressed_mrr']:.4f}\n")
                    f.write(f"MRR increase amount: {item['mrr_diff']:.4f}\n")
                    f.write(f"MRR increase rate: {item['mrr_diff_rate']:.2f}%\n")
                    f.write(f"Context: {file1_sample['context']}\n")
                    f.write(f"File 1 increase rate: {file1_sample['mrr_change_rate']}\n")
                    f.write(f"File 2 increase rate: {file2_sample['mrr_change_rate']}\n")
                    f.write("-" * 80 + "\n\n")
            else:
                f.write("No samples with same text and higher MRR in comparison file\n\n")

            f.write("4. Same Text Samples with Lower Suppressed MRR in Comparison File\n")
            f.write("-" * 50 + "\n")
            if same_sample_worse_mrr:
                f.write(f"Total {len(same_sample_worse_mrr)} samples (sorted by MRR decrease rate):\n\n")
                for idx, item in enumerate(same_sample_worse_mrr, 1):
                    file1_sample = item["file1_sample"]
                    file2_sample = item["file2_sample"]

                    f.write(f"【Decreased Sample {idx}】\n")
                    f.write(f"Original line: {file1_sample['original_line']}\n")
                    f.write(f"Target name: {file1_sample['target_name']}\n")
                    f.write(f"File 1 suppressed MRR: {file1_sample['suppressed_mrr']:.4f}\n")
                    f.write(f"File 2 suppressed MRR: {file2_sample['suppressed_mrr']:.4f}\n")
                    f.write(f"MRR decrease amount: {abs(item['mrr_diff']):.4f}\n")
                    f.write(f"MRR decrease rate: {abs(item['mrr_diff_rate']):.2f}%\n")
                    f.write(f"Context: {file1_sample['context']}\n")
                    f.write(f"File 1 increase rate: {file1_sample['mrr_change_rate']}\n")
                    f.write(f"File 2 increase rate: {file2_sample['mrr_change_rate']}\n")
                    f.write("-" * 80 + "\n\n")
            else:
                f.write("No samples with same text and lower MRR in comparison file\n")

        print("\n【Preview of unique samples in comparison file (first 5)】")
        for idx, sample in enumerate(file2_unique_samples[:5], 1):
            print(f"{idx}. Target name: {sample['target_name']} | Suppressed MRR: {sample['suppressed_mrr']:.4f}")
            print(f"   Context: {sample['context'][:60]}...")
            print()

        print("\n【Preview of same text samples with increased MRR (first 5)】")
        for idx, item in enumerate(same_sample_better_mrr[:5], 1):
            file1_sample = item["file1_sample"]
            file2_sample = item["file2_sample"]
            print(f"{idx}. Target name: {file1_sample['target_name']}")
            print(f"   File 1 MRR: {file1_sample['suppressed_mrr']:.4f} → File 2 MRR: {file2_sample['suppressed_mrr']:.4f}")
            print(f"   Increase amount: {item['mrr_diff']:.4f} ({item['mrr_diff_rate']:.2f}%)")
            print()

        print("\n【Preview of same text samples with decreased MRR (first 5)】")
        for idx, item in enumerate(same_sample_worse_mrr[:5], 1):
            file1_sample = item["file1_sample"]
            file2_sample = item["file2_sample"]
            print(f"{idx}. Target name: {file1_sample['target_name']}")
            print(f"   File 1 MRR: {file1_sample['suppressed_mrr']:.4f} → File 2 MRR: {file2_sample['suppressed_mrr']:.4f}")
            print(f"   Decrease amount: {abs(item['mrr_diff']):.4f} ({abs(item['mrr_diff_rate']):.2f}%)")
            print()

    def generate_mrr_increase_histogram(self, top_samples: List[Dict], output_path: str):
        if not top_samples:
            print("No samples found for generating increase histogram")
            return

        sample_labels = []
        pre_pane_mrr = []
        post_pane_mrr = []

        for idx, item in enumerate(top_samples, 1):
            file1_sample = item["file1_sample"]
            file2_sample = item["file2_sample"]

            sample_labels.append(f"Sample {idx}\n(+{item['mrr_diff_rate']:.1f}%)")

            pre_pane_mrr.append(file1_sample["suppressed_mrr"])
            post_pane_mrr.append(file2_sample["suppressed_mrr"])

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(sample_labels))
        width = 0.35

        bars1 = ax.bar(x - width / 2, pre_pane_mrr, width,
                       label='Pre-edit MRR', color='#1f77b4', alpha=0.9)
        bars2 = ax.bar(x + width / 2, post_pane_mrr, width,
                       label='Post-edit MRR', color='#ff7f0e', alpha=0.9)

        ax.set_title('Samples MRR Before & After Editing', fontsize=18, fontweight='bold', pad=25)
        ax.set_xlabel('Samples (MRR Increase Rate in Parentheses)', fontsize=14)
        ax.set_ylabel('Suppressed MRR Value', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(sample_labels, rotation=0, fontsize=10)

        self._add_bar_labels(bars1, ax)
        self._add_bar_labels(bars2, ax)

        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

        min_mrr = min(min(pre_pane_mrr), min(post_pane_mrr)) * 0.95
        max_mrr = max(max(pre_pane_mrr), max(post_pane_mrr)) * 1.05
        ax.set_ylim(min_mrr, max_mrr)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
        plt.close()

        print(f"\nMRR increase sample histogram saved to: {os.path.abspath(output_path)}")

    def generate_mrr_decrease_histogram(self, top_samples: List[Dict], output_path: str):
        if not top_samples:
            print("No samples found for generating decrease histogram")
            return

        sample_labels = []
        pre_pane_mrr = []
        post_pane_mrr = []

        for idx, item in enumerate(top_samples, 1):
            file1_sample = item["file1_sample"]
            file2_sample = item["file2_sample"]

            sample_labels.append(f"Sample {idx}\n(-{abs(item['mrr_diff_rate']):.1f}%)")

            pre_pane_mrr.append(file1_sample["suppressed_mrr"])
            post_pane_mrr.append(file2_sample["suppressed_mrr"])

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(sample_labels))
        width = 0.35

        bars1 = ax.bar(x - width / 2, pre_pane_mrr, width,
                       label='Pre-edit MRR', color='#1f77b4', alpha=0.9)
        bars2 = ax.bar(x + width / 2, post_pane_mrr, width,
                       label='Post-PANE MRR', color='#2ca02c', alpha=0.9)

        ax.set_title('Samples MRR Before & After Editing ', fontsize=18, fontweight='bold', pad=25)
        ax.set_xlabel('Samples (MRR Decrease Rate in Parentheses)', fontsize=14)
        ax.set_ylabel('Suppressed MRR Value', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(sample_labels, rotation=0, fontsize=10)

        self._add_bar_labels(bars1, ax)
        self._add_bar_labels(bars2, ax)

        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

        min_mrr = min(min(pre_pane_mrr), min(post_pane_mrr)) * 0.95
        max_mrr = max(max(pre_pane_mrr), max(post_pane_mrr)) * 1.05
        ax.set_ylim(min_mrr, max_mrr)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
        plt.close()

        print(f"MRR decrease sample histogram saved to: {os.path.abspath(output_path)}")

    def _add_bar_labels(self, bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, fontweight='500')

    @staticmethod
    def get_current_time():
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    config = {
        "file1_path": r"D:\expt\lrp_spin\main\Seesaw\sample_seesaw\mrr_increased_samples_name.txt",
        "file2_path": r"D:\expt\lrp_spin\main\Seesaw\sample_seesaw\mrr_increased_samples_phone.txt",
        "output_file": r"D:\expt\lrp_spin\main\Seesaw\sample_seesaw\mrr_samples_comparison_report.txt",
        "plot_increase_output": r"D:\expt\lrp_spin\main\Seesaw\mrr_increase_histogram.png",
        "plot_decrease_output": r"D:\expt\lrp_spin\main\Seesaw\mrr_decrease_histogram.png"
    }

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib library not detected, attempting to install automatically...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        print("matplotlib installed successfully!")

    comparator = MRRIncreasedSamplesComparator()
    comparator.compare_samples(
        file1_path=config["file1_path"],
        file2_path=config["file2_path"],
        output_file=config["output_file"],
        plot_increase_output=config["plot_increase_output"],
        plot_decrease_output=config["plot_decrease_output"]
    )


if __name__ == "__main__":
    main()