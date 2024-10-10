import csv
import numpy as np
import torch
import pandas as pd
import random

TEST_DATA = "data/anlp01-sts-free-test.tsv"


class RandomModel:
    def __init__(self):
        self.model = None

    def predict(self, data):
        return torch.tensor([random.uniform(0, 6) for _ in range(data.shape[0])])  # random float between 0 and 6
        # return torch.randint(0, 7, (data.shape[0],), dtype=torch.float32)


class MajorityModel:
    def __init__(self, best_prob):
        self.best_prob = best_prob
        self.model = None

    def predict(self, data):
        return torch.full((data.shape[0],), self.best_prob)


test_data = pd.read_csv(TEST_DATA, sep='\t', header=None, encoding="utf-8", quoting=csv.QUOTE_NONE, quotechar='"')
test_data = test_data[2]
test_data = np.round(test_data, 0)

random_model = RandomModel()
majority_model = MajorityModel(0.0)  # majority class is 0 in the training data
majority_predictions = majority_model.predict(torch.tensor(test_data, dtype=torch.float32))

random_loss_mse = 0
random_loss_mae = 0
for i in range(30):
    random_predictions = random_model.predict(torch.tensor(test_data, dtype=torch.float32))
    random_loss_mse += torch.nn.functional.mse_loss(random_predictions, torch.tensor(test_data, dtype=torch.float32))
    random_loss_mae += torch.nn.functional.l1_loss(random_predictions, torch.tensor(test_data, dtype=torch.float32))
random_loss_mse /= 30
random_loss_mae /= 30

majority_loss_mse = torch.nn.functional.mse_loss(majority_predictions, torch.tensor(test_data, dtype=torch.float32))
majority_loss_mae = torch.nn.functional.l1_loss(majority_predictions, torch.tensor(test_data, dtype=torch.float32))

random_mse_loss = random_loss_mse.item()
random_mae_loss = random_loss_mae.item()
majority_mse_loss = majority_loss_mse.item()
majority_mae_loss = majority_loss_mae.item()

loss_data = {
    "Model": ["Random", "Random", "Majority", "Majority"],
    "Loss Type": ["MSE", "MAE", "MSE", "MAE"],
    "Loss Value": [random_mse_loss, random_mae_loss, majority_mse_loss, majority_mae_loss]
}

loss_df = pd.DataFrame(loss_data)
print(loss_df.to_markdown(index=False))
