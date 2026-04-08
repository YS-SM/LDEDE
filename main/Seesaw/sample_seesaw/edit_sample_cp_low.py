import os
import re
from typing import Dict, List, Set
from collections import defaultdict
from datetime import datetime


class MRRIncreasedSamplesComparator:
    def __init__(self):
        self.sample_pattern = re.compile(
            r"\[Sample\d+\]\s+"
            r"Original line：(context: .*? \| privacy: .*? \| MRR: [\d\.]+)\s+"
            r"MRR after suppression：([\d\.]+)\s+"
            r"MRR change amount：([\d\.]+)\s+"
            r"MRR increase rate：([\d\.]+%)\s+"
            r"Context：(.*?)\s+"
            r"Target name：(.*?)\s+"
            r"-{5,}",
            re.DOTALL
        )
        self.sample_key_func = lambda ctx, target: (ctx.strip(), target.strip())

    def parse_sample_file(self, file_path: str) -> Dict[tuple, Dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")

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

    def compare_samples(self, file1_path: str, file2_path: str, output_file: str = "mrr_samples_comparison_report.txt"):
        print("=" * 80)
        print("Start comparing MRR increased sample files")
        print(f"Baseline file (File 1)：{file1_path}")
        print(f"Comparison file (File 2)：{file2_path}")
        print("Core objective：Find samples where MRR in File 1 < MRR in File 2 for the same sample")
        print("=" * 80)

        try:
            file1_samples = self.parse_sample_file(file1_path)
            file2_samples = self.parse_sample_file(file2_path)
        except Exception as e:
            print(f"Failed to parse files：{str(e)}")
            return

        file1_lower_mrr_samples = []
        common_sample_keys = set(file1_samples.keys()) & set(file2_samples.keys())

        for sample_key in common_sample_keys:
            file1_sample = file1_samples[sample_key]
            file2_sample = file2_samples[sample_key]

            if file1_sample["suppressed_mrr"] < file2_sample["suppressed_mrr"]:
                mrr_diff = file2_sample["suppressed_mrr"] - file1_sample["suppressed_mrr"]
                mrr_diff_rate = (mrr_diff / file1_sample["suppressed_mrr"]) * 100 if file1_sample["suppressed_mrr"] != 0 else 0.0
                file1_lower_mrr_samples.append({
                    "file1_sample": file1_sample,
                    "file2_sample": file2_sample,
                    "mrr_diff": mrr_diff,
                    "mrr_diff_rate": mrr_diff_rate
                })

        file1_lower_mrr_samples.sort(key=lambda x: x["mrr_diff_rate"], reverse=True)

        self.generate_report(
            file1_samples=file1_samples,
            file2_samples=file2_samples,
            file1_lower_mrr_samples=file1_lower_mrr_samples,
            common_sample_keys=common_sample_keys,
            output_file=output_file,
            file1_path=file1_path,
            file2_path=file2_path
        )

        print("\n" + "=" * 80)
        print("Comparison completed!")
        print(f"Comparison report saved to：{os.path.abspath(output_file)}")
        print(f"Statistics：")
        print(f"- Total samples in baseline file (File 1)：{len(file1_samples)}")
        print(f"- Total samples in comparison file (File 2)：{len(file2_samples)}")
        print(f"- Number of common samples in both files：{len(common_sample_keys)}")
        print(f"- Number of core target samples (File 1 MRR < File 2 MRR)：{len(file1_lower_mrr_samples)}")
        if common_sample_keys:
            print(f"- Proportion of core target samples in common samples：{len(file1_lower_mrr_samples) / len(common_sample_keys) * 100:.2f}%")
        print("=" * 80)

    def generate_report(self, file1_samples: Dict, file2_samples: Dict,
                        file1_lower_mrr_samples: List[Dict], common_sample_keys: Set[tuple],
                        output_file: str, file1_path: str, file2_path: str):
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("MRR Increased Sample Files Comparison Report (Core: Samples with File 1 MRR < File 2 MRR)\n")
            f.write("=" * 100 + "\n")
            f.write(f"Generation time：{self.get_current_time()}\n")
            f.write(f"Baseline file (File 1)：{os.path.basename(file1_path)}\n")
            f.write(f"Comparison file (File 2)：{os.path.basename(file2_path)}\n")
            f.write(f"Baseline file path：{file1_path}\n")
            f.write(f"Comparison file path：{file2_path}\n")
            f.write("\n")

            f.write("1. Overall Statistics\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total samples in baseline file (File 1)：{len(file1_samples)}\n")
            f.write(f"Total samples in comparison file (File 2)：{len(file2_samples)}\n")
            f.write(f"Number of common samples in both files：{len(common_sample_keys)}\n")
            f.write(f"Number of core target samples (File 1 MRR < File 2 MRR)：{len(file1_lower_mrr_samples)}\n")
            if common_sample_keys:
                f.write(
                    f"Proportion of core target samples in common samples：{len(file1_lower_mrr_samples) / len(common_sample_keys) * 100:.2f}%\n")
            f.write("\n")

            f.write("2. Core Target Samples Details (File 1 MRR < File 2 MRR, sorted by increase rate)\n")
            f.write("-" * 50 + "\n")
            if file1_lower_mrr_samples:
                f.write(f"Total {len(file1_lower_mrr_samples)} samples：\n\n")
                for idx, item in enumerate(file1_lower_mrr_samples, 1):
                    file1_sample = item["file1_sample"]
                    file2_sample = item["file2_sample"]

                    f.write(f"[Core Sample {idx}]\n")
                    f.write(f"Original line：{file1_sample['original_line']}\n")
                    f.write(f"Target name：{file1_sample['target_name']}\n")
                    f.write(f"File 1 MRR after suppression：{file1_sample['suppressed_mrr']:.4f}\n")
                    f.write(f"File 2 MRR after suppression：{file2_sample['suppressed_mrr']:.4f}\n")
                    f.write(f"MRR increase amount：{item['mrr_diff']:.4f}\n")
                    f.write(f"MRR increase rate (relative to File 1)：{item['mrr_diff_rate']:.2f}%\n")
                    f.write(f"Context：{file1_sample['context']}\n")
                    f.write(f"File 1 MRR change amount：{file1_sample['mrr_change']:.4f}\n")
                    f.write(f"File 1 MRR increase rate：{file1_sample['mrr_change_rate']}\n")
                    f.write(f"File 2 MRR change amount：{file2_sample['mrr_change']:.4f}\n")
                    f.write(f"File 2 MRR increase rate：{file2_sample['mrr_change_rate']}\n")
                    f.write("-" * 80 + "\n\n")
            else:
                f.write("No samples found where File 1 MRR < File 2 MRR for the same sample\n")

            f.write("3. Auxiliary Statistics\n")
            f.write("-" * 50 + "\n")
            file1_unique_keys = set(file1_samples.keys()) - common_sample_keys
            f.write(f"Number of unique samples in File 1：{len(file1_unique_keys)}\n")
            file2_unique_keys = set(file2_samples.keys()) - common_sample_keys
            f.write(f"Number of unique samples in File 2：{len(file2_unique_keys)}\n")
            file1_higher_eq_mrr_count = len(common_sample_keys) - len(file1_lower_mrr_samples)
            f.write(f"Number of common samples where File 1 MRR >= File 2 MRR：{file1_higher_eq_mrr_count}\n")

        print("\n[Core Target Samples Preview (Top 10, sorted by increase rate)]")
        for idx, item in enumerate(file1_lower_mrr_samples[:10], 1):
            file1_sample = item["file1_sample"]
            file2_sample = item["file2_sample"]
            print(f"{idx}. Target name：{file1_sample['target_name']}")
            print(f"   File 1 MRR：{file1_sample['suppressed_mrr']:.4f} → File 2 MRR：{file2_sample['suppressed_mrr']:.4f}")
            print(f"   Increase amount：{item['mrr_diff']:.4f} (Increase rate：{item['mrr_diff_rate']:.2f}%)")
            print(f"   Context：{file1_sample['context'][:80]}...")
            print("-" * 60)

    @staticmethod
    def get_current_time():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    config = {
        "file1_path": r"D:\expt\lrp_spin\main\Seesaw\sample_seesaw\mrr_increased_samples_phone_wutarge.txt",
        "file2_path": r"D:\expt\lrp_spin\main\Seesaw\sample_seesaw\mrr_increased_samples_phone.txt",
        "output_file": r"D:\expt\lrp_spin\main\Seesaw\sample_seesaw\mrr_samples_comparison_report_file1_lower_mrr.txt"
    }

    comparator = MRRIncreasedSamplesComparator()
    comparator.compare_samples(
        file1_path=config["file1_path"],
        file2_path=config["file2_path"],
        output_file=config["output_file"]
    )


if __name__ == "__main__":
    main()