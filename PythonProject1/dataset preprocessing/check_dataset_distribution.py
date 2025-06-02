import numpy as np

# ✅ Load Tumor Stage Labels
labels_path = "D:/pycharm/tumor_stages.npy"
labels = np.load(labels_path, allow_pickle=True).item()

# ✅ Count the number of samples per stage
stage_counts = {0: 0, 1: 0, 2: 0, 3: 0}
for stage in labels.values():
    stage_counts[stage] += 1

# ✅ Print stage distribution
print("📊 Tumor Stage Distribution:")
for stage, count in stage_counts.items():
    print(f"Grade {stage + 1}: {count} samples")
