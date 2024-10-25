import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

CSFD_DATASET_TRAIN = "data/csfd-train.tsv"
CSFD_DATASET_TEST = "data/csfd-test.tsv"

train_data = pd.read_csv(CSFD_DATASET_TRAIN, sep='\t', encoding="utf-8", quoting=csv.QUOTE_NONE, quotechar='"')
test_data = pd.read_csv(CSFD_DATASET_TEST, sep='\t', encoding="utf-8", quoting=csv.QUOTE_NONE, quotechar='"')

# Create histogram of pair similarity.
train_sim_hist = torch.histc(torch.tensor(train_data["label"], dtype=torch.float32), bins=3)
test_sim_hist = torch.histc(torch.tensor(np.round(test_data["label"], 0), dtype=torch.float32), bins=3)


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
    # plt.savefig(f"img/{title}.svg")
    plt.show()


plot_histogram(train_sim_hist, "Train Sentiment Score Histogram", 'skyblue', 'Train')
plot_histogram(test_sim_hist, "Test Sentiment Score Histogram", 'lime', 'Test')

# Present mean and std of the dataset.
train_mean = train_data["label"].mean()
train_std = train_data["label"].std()
test_mean = test_data["label"].mean()
test_std = test_data["label"].std()

# create a table as a pandas DataFrame
data = {'Dataset': ['Train', 'Test'],
        'Mean': [train_mean, test_mean],
        'Std': [train_std, test_std]}
df = pd.DataFrame(data)
print(df.to_markdown(index=False))

# Text length analysis
# Create histogram of text length.
train_text_hist = train_data["text"].apply(lambda x: len(x.split())).value_counts().sort_index()

plt.figure(figsize=(10, 8))
plt.bar(train_text_hist.index, train_text_hist, color='skyblue', label='Train')
plt.ylabel("Count", fontsize=14)
plt.xlabel("Text Length", fontsize=14)
plt.title("Train Text Length Histogram", fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"img/Train Text Length Histogram.svg")
plt.show()



train_text_avg = train_data["text"].apply(lambda x: len(x.split())).mean()
train_text_std = train_data["text"].apply(lambda x: len(x.split())).std()
train_text_q = train_data["text"].apply(lambda x: len(x.split())).quantile(0.88)

print(f"Average text length in train dataset: {train_text_avg} +- {train_text_std}")
print(f"88% quantile text length in train dataset: {train_text_q}")
