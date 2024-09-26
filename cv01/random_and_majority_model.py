import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class RandomModel:
    def __init__(self):
        self.model = None

    def fit(self, train_data, train_targets):
        pass

    def predict(self, data):
        return torch.randint(0, 10, (data.shape[0],))

class MajorityModel:
    def __init__(self, majority_class):
        self.majority_class = majority_class
        self.model = None

    def predict(self, data):
        return torch.full((data.shape[0],), self.majority_class)

# def test(model, true_labels):



transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('data', train=False, transform=transform)

train_targets = dataset1.targets
majority_class = train_targets.mode().values.item()

random_model = RandomModel()
majority_model = MajorityModel(majority_class)

random_predictions = random_model.predict(dataset2.data)
majority_predictions = majority_model.predict(dataset2.data)


print("Random model accuracy:", (random_predictions == dataset2.targets).float().mean().item())
print("Majority model accuracy:", (majority_predictions == dataset2.targets).float().mean().item())
print("Number off class 1 occurences in the test dataset:", (dataset2.targets == 1).sum().item())
print("Total number of samples in the test dataset:", len(dataset2.targets))



