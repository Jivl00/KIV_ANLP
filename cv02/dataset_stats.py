import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

TRAIN_DATA = "data/anlp01-sts-free-train.tsv"
TEST_DATA = "data/anlp01-sts-free-test.tsv"

train_data = pd.read_csv(TRAIN_DATA, sep='\t', header=None, encoding="utf-8", quoting=csv.QUOTE_NONE, quotechar='"')
test_data = pd.read_csv(TEST_DATA, sep='\t', header=None, encoding="utf-8", quoting=csv.QUOTE_NONE, quotechar='"')
print(len(train_data), len(test_data))

# Create histogram of pair similarity.
train_sim_hist = torch.histc(torch.tensor(train_data[2], dtype=torch.float32), bins=7)
test_sim_hist = torch.histc(torch.tensor(np.round(test_data[2], 0), dtype=torch.float32), bins=7)


def plot_histogram(data, title, color, label):
    plt.figure(figsize=(10, 8))
    plt.bar(range(len(data)), data, color=color, edgecolor='black', label=label)
    for i, v in enumerate(data):
        plt.text(i, v + 0.5, str(int(v.item())), ha='center', va='top', fontsize=12, rotation=90)
    plt.ylabel("Count", fontsize=14)
    plt.xlabel("Similarity", fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(range(len(data)), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"img/{title}.svg")
    plt.show()


plot_histogram(train_sim_hist, "Train Pair Similarity Histogram", 'skyblue', 'Train')
plot_histogram(test_sim_hist, "Test Pair Similarity Histogram", 'lime', 'Test')

# Present mean and std of the dataset.
train_mean = train_data[2].mean()
train_std = train_data[2].std()
test_mean = test_data[2].mean()
test_std = test_data[2].std()

# create a table as a pandas DataFrame
data = {'Dataset': ['Train', 'Test'],
        'Mean': [train_mean, test_mean],
        'Std': [train_std, test_std]}
df = pd.DataFrame(data)
print(df.to_markdown(index=False))