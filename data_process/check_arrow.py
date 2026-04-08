from datasets import load_from_disk

dataset_path = r"D:\expt\lrp_spin\data_process\enron_arrow\valid\final"
dataset = load_from_disk(dataset_path)

print("="*50)
print("Basic information of the dataset:")
print(f"Number of samples: {len(dataset)}")
print(f"Included columns: {dataset.column_names}")
print(f"Feature description: {dataset.features}")
print("="*50)

num_samples = 5
print(f"\nFirst {num_samples} samples content:")
for i in range(num_samples):
    print(f"\n--- Sample {i+1} ---")
    sample = dataset[i]
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")