from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('data', train=False, transform=transform)

targets = torch.cat((dataset1.targets, dataset2.targets))
# print(len(targets))
# class_1 = 0
# for i in range(len(targets)):
#     if targets[i] == 1:
#         class_1 += 1
# print(class_1)

class_hist = torch.bincount(targets)
plt.figure(figsize=(10, 6))
plt.bar(range(len(class_hist)), class_hist, color='skyblue', edgecolor='black')
for i, v in enumerate(class_hist):
    plt.text(i, v + 0.5, str(v.item()), ha='center', va='top', fontsize=12, rotation=90)
    plt.xlabel("Class", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.title("Class Histogram", fontsize=16)
plt.xticks(range(len(class_hist)), fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
