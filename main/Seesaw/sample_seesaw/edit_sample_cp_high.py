import os
import re
from typing import Dict, List, Set
from collections import defaultdict
from datetime import datetime


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

    def compare_samples(self, file1_path: str, file2_path: str, output_file: str = "mrr_samples_comparison_report.txt"):
        print("=" * 80)
        print("Start comparing MRR increased sample files")
        print(f"Baseline file：{file1_path}")
        print(f"Comparison file：{file2_path}")
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

        same_sample_better_mrr.sort(key=lambda x: x["mrr_diff_rate"], reverse=True)

        self.generate_report(
            file1_samples=file1_samples,
            file2_samples=file2_samples,
            file2_unique_samples=file2_unique_samples,
            same_sample_better_mrr=same_sample_better_mrr,
            output_file=output_file,
            file1_path=file1_path,
            file2_path=file2_path
        )

        print("\n" + "=" * 80)
        print("Comparison completed!")
        print(f"Comparison report saved to: {os.path.abspath(output_file)}")
        print(f"Statistics：")
        print(f"- Total samples in baseline file: {len(file1_samples)}")
        print(f"- Total samples in comparison file: {len(file2_samples)}")
        print(f"- Unique samples in comparison file: {len(file2_unique_samples)}")
        print(f"- Samples with higher MRR in comparison file (same text): {len(same_sample_better_mrr)}")
        print("=" * 80)

    def generate_report(self, file1_samples: Dict, file2_samples: Dict,
                        file2_unique_samples: List[Dict], same_sample_better_mrr: List[Dict],
                        output_file: str, file1_path: str, file2_path: str):
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("MRR Increased Samples Comparison Report\n")
            f.write("=" * 100 + "\n")
            f.write(f"Generation time：{self.get_current_time()}\n")
            f.write(f"Baseline file (File 1)：{os.path.basename(file1_path)}\n")
            f.write(f"Comparison file (File 2)：{os.path.basename(file2_path)}\n")
            f.write(f"Baseline file path：{file1_path}\n")
            f.write(f"Comparison file path：{file2_path}\n")
            f.write("\n")

            f.write("1. Overall Statistics\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total samples in baseline file: {len(file1_samples)}\n")
            f.write(f"Total samples in comparison file: {len(file2_samples)}\n")
            f.write(f"Unique samples in comparison file: {len(file2_unique_samples)}\n")
            f.write(f"Samples with higher MRR in comparison file (same text): {len(same_sample_better_mrr)}\n")
            f.write(f"Common samples in both files: {len(set(file1_samples.keys()) & set(file2_samples.keys()))}\n")
            f.write("\n")

            f.write("2. Unique Samples in Comparison File (Not in File 1)\n")
            f.write("-" * 50 + "\n")
            if file2_unique_samples:
                f.write(f"Total {len(file2_unique_samples)} unique samples：\n\n")
                for idx, sample in enumerate(file2_unique_samples, 1):
                    f.write(f"【Unique Sample {idx}】\n")
                    f.write(f"Original line：{sample['original_line']}\n")
                    f.write(f"Suppressed MRR：{sample['suppressed_mrr']:.4f}\n")
                    f.write(f"MRR change：{sample['mrr_change']:.4f}\n")
                    f.write(f"MRR increase rate：{sample['mrr_change_rate']}\n")
                    f.write(f"Context：{sample['context']}\n")
                    f.write(f"Target name：{sample['target_name']}\n")
                    f.write("-" * 80 + "\n\n")
            else:
                f.write("No unique samples in comparison file (all samples exist in File 1)\n\n")

            f.write("3. Samples with Same Text but Higher Suppressed MRR in Comparison File\n")
            f.write("-" * 50 + "\n")
            if same_sample_better_mrr:
                f.write(f"Total {len(same_sample_better_mrr)} samples (sorted by MRR increase rate)：\n\n")
                for idx, item in enumerate(same_sample_better_mrr, 1):
                    file1_sample = item["file1_sample"]
                    file2_sample = item["file2_sample"]

                    f.write(f"【Improved Sample {idx}】\n")
                    f.write(f"Original line：{file1_sample['original_line']}\n")
                    f.write(f"Target name：{file1_sample['target_name']}\n")
                    f.write(f"File 1 suppressed MRR：{file1_sample['suppressed_mrr']:.4f}\n")
                    f.write(f"File 2 suppressed MRR：{file2_sample['suppressed_mrr']:.4f}\n")
                    f.write(f"MRR improvement：{item['mrr_diff']:.4f}\n")
                    f.write(f"MRR improvement rate：{item['mrr_diff_rate']:.2f}%\n")
                    f.write(f"Context：{file1_sample['context']}\n")
                    f.write(f"File 1 increase rate：{file1_sample['mrr_change_rate']}\n")
                    f.write(f"File 2 increase rate：{file2_sample['mrr_change_rate']}\n")
                    f.write("-" * 80 + "\n\n")
            else:
                f.write("No samples with same text and higher MRR in comparison file\n")

        print("\n【Preview of unique samples in comparison file (first 5)】")
        for idx, sample in enumerate(file2_unique_samples[:5], 1):
            print(f"{idx}. Target name：{sample['target_name']} | Suppressed MRR：{sample['suppressed_mrr']:.4f}")
            print(f"   Context：{sample['context'][:60]}...")
            print()

        print("\n【Preview of MRR improved samples with same text (first 5)】")
        for idx, item in enumerate(same_sample_better_mrr[:5], 1):
            file1_sample = item["file1_sample"]
            file2_sample = item["file2_sample"]
            print(f"{idx}. Target name：{file1_sample['target_name']}")
            print(f"   File 1 MRR：{file1_sample['suppressed_mrr']:.4f} → File 2 MRR：{file2_sample['suppressed_mrr']:.4f}")
            print(f"   Improvement：{item['mrr_diff']:.4f}（{item['mrr_diff_rate']:.2f}%）")
            print()

    @staticmethod
    def get_current_time():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    config = {
        "file1_path": r"D:\expt\lrp_spin\main\Seesaw\sample_seesaw\mrr_increased_samples_name.txt",
        "file2_path": r"D:\expt\lrp_spin\main\Seesaw\sample_seesaw\mrr_increased_samples_phone.txt",
        "output_file": r"D:\expt\lrp_spin\main\Seesaw\sample_seesaw\mrr_samples_comparison_report.txt"
    }

    comparator = MRRIncreasedSamplesComparator()
    comparator.compare_samples(
        file1_path=config["file1_path"],
        file2_path=config["file2_path"],
        output_file=config["output_file"]
    )


if __name__ == "__main__":
    main()