import wandb
import sys

sys.path.append("../")
from wandb_config import WANDB_PROJECT, WANDB_ENTITY
import numpy as np
import pandas as pd

api = wandb.Api()
# all_runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
# print(len(all_runs))

# dense = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", {"$and": [{"config.model": "dense", "state": "finished"}]})
# cnn = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", {"$and": [{"config.model": "cnn", "state": "finished"}]})
# print(len(dense), len(cnn))
#
# # Get 2 best HP configuration for dense and cnn model
# dense_hp = []
# cnn_hp = []
# for run in dense:
#     if "test_acc" not in run.summary:
#         continue
#     if run.summary["test_acc"] > 99:
#         dense_hp.append((run.summary["test_acc"], run.config))
# for run in cnn:
#     if "test_acc" not in run.summary:
#         continue
#     if run.summary["test_acc"] > 99.9:
#         cnn_hp.append((run.summary["test_acc"], run.config))
#
# print(len(dense_hp), len(cnn_hp))
# print(dense_hp)
# print(cnn_hp)

best_dense_hp1 = {'dp': 0, 'lr': 0.001, 'model': 'dense', 'optimizer': 'adam'}
best_dense_hp2 = {'dp': 0.3, 'lr': 0.001, 'model': 'dense', 'optimizer': 'adam'}

best_cnn_hp1 = {'dp': 0, 'lr': 0.0001, 'model': 'cnn', 'optimizer': 'adam'}
best_cnn_hp2 = {'dp': 0.5, 'lr': 0.001, 'model': 'cnn', 'optimizer': 'adam'}

bests = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", {"$and": [
    {"config.model": "dense", "state": "finished", "config.dp": 0, "config.lr": 0.001, "config.optimizer": "adam"}]})
best_dense_accs1 = [run.summary["test_acc"] for run in bests]
bests = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", {"$and": [
    {"config.model": "dense", "state": "finished", "config.dp": 0.3, "config.lr": 0.001, "config.optimizer": "adam"}]})
best_dense_accs2 = [run.summary["test_acc"] for run in bests]

bests = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", {"$and": [
    {"config.model": "cnn", "state": "finished", "config.dp": 0, "config.lr": 0.0001, "config.optimizer": "adam"}]})
best_cnn_accs1 = [run.summary["test_acc"] for run in bests]
bests = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", {"$and": [
    {"config.model": "cnn", "state": "finished", "config.dp": 0.5, "config.lr": 0.001, "config.optimizer": "adam"}]})
best_cnn_accs2 = [run.summary["test_acc"] for run in bests]

print(len(best_dense_accs1), len(best_dense_accs2), len(best_cnn_accs1), len(best_cnn_accs2))

# Create table
data = {
    "model": ["dense", "dense", "cnn", "cnn", "random", "majority"],
    "dp": [0, 0.3, 0, 0.5, None, None],
    "lr": [0.001, 0.001, 0.0001, 0.001, None, None],
    "optimizer": ["adam", "adam", "adam", "adam", None, None],
    # mean +- confidence interval
    "accuracy -+ 95% confidence": [f"{np.mean(best_dense_accs1):.2f} +- {1.96 * np.std(best_dense_accs1) / np.sqrt(len(best_dense_accs1)):.2f}",
                                      f"{np.mean(best_dense_accs2):.2f} +- {1.96 * np.std(best_dense_accs2) / np.sqrt(len(best_dense_accs2)):.2f}",
                                      f"{np.mean(best_cnn_accs1):.2f} +- {1.96 * np.std(best_cnn_accs1) / np.sqrt(len(best_cnn_accs1)):.2f}",
                                      f"{np.mean(best_cnn_accs2):.2f} +- {1.96 * np.std(best_cnn_accs2) / np.sqrt(len(best_cnn_accs2)):.2f}",
                 "cca 0.10", "cca 0.10"]
}
df = pd.DataFrame(data)
# make last column bold
df.iloc[-2:, -1] = df.iloc[-2:, -1].apply(lambda x: f"**{x}**")
print(df.to_markdown(index=False))
