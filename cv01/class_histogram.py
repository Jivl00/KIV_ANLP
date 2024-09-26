from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('data', train=False, transform=transform)

targets1 = dataset1.targets
targets2 = dataset2.targets

class_hist1 = torch.bincount(targets1, minlength=10)
class_hist2 = torch.bincount(targets2, minlength=10)

plt.figure(figsize=(10, 8))
plt.bar(range(len(class_hist1)), class_hist1, color='skyblue', edgecolor='black', label='Train')
plt.bar(range(len(class_hist2)), class_hist2, color='lime', edgecolor='black', bottom=class_hist1, label='Test')
for i, (v1, v2) in enumerate(zip(class_hist1, class_hist2)):
    plt.text(i, v1 + 0.5, str(v1.item()), ha='center', va='top', fontsize=12, rotation=90)
    plt.text(i, v1 + v2 + 0.5, str(v2.item()), ha='center', va='top', fontsize=12, rotation=90)
plt.ylabel("Count", fontsize=14)
plt.title("Class Histogram", fontsize=16)
plt.xticks(range(len(class_hist1)), fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='lower right')
plt.show()
