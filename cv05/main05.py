import csv

import pandas as pd
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
import argparse

WANDB_PROJECT = "anlp-2024_kimlova_vladimira"
WANDB_ENTITY = "anlp2024"


class RegressionModel(nn.Module):
    def __init__(self, base_model):
        super(RegressionModel, self).__init__()
        self.base_model = base_model
        self.regressor = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.regressor(pooled_output)


def main(config):
    # wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=["cv05", "best"], config=config)
    # wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=["cv05"], config=config)
    # if config["task"] == "sts":
    #     wandb.log({"test_loss": None, "train_loss": None})
    # if config["task"] == "sentiment":
    #     wandb.log({"test_loss": None, "test_acc": None, "train_loss": None})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config["model_type"])
    if config["task"] == "sts":
        base_model = AutoModel.from_pretrained(config["model_type"])
        model = RegressionModel(base_model)
        model.to(device)

        TRAIN_DATA = "data-sts/anlp01-sts-free-train.tsv"
        TEST_DATA = "data-sts/anlp01-sts-free-test.tsv"
        train_data = pd.read_csv(TRAIN_DATA, sep='\t', header=None, encoding="utf-8", quoting=csv.QUOTE_NONE,
                                 quotechar='"')
        test_data = pd.read_csv(TEST_DATA, sep='\t', header=None, encoding="utf-8", quoting=csv.QUOTE_NONE,
                                quotechar='"')

        # train_texts = [f"{s1} {s2}" for s1, s2 in zip(train_data[0], train_data[1])]
        train_labels = torch.tensor(train_data[2].values, dtype=torch.float32).unsqueeze(1)
        # test_texts = [f"{s1} {s2}" for s1, s2 in zip(test_data[0], test_data[1])]
        test_labels = torch.tensor(test_data[2].values, dtype=torch.float32).unsqueeze(1)
        #
        # train_encodings = tokenizer(train_texts, return_tensors="pt", padding=True, truncation=True, max_length=100)
        # test_encodings = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=100)
        #
        # train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
        # test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

        # Tokenize each sentence separately
        train_encodings_1 = tokenizer(train_data[0].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=100)
        train_encodings_2 = tokenizer(train_data[1].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=100)
        test_encodings_1 = tokenizer(test_data[0].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=100)
        test_encodings_2 = tokenizer(test_data[1].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=100)

        # Concatenate the tokenized sentences
        train_input_ids = torch.cat((train_encodings_1['input_ids'], train_encodings_2['input_ids']), dim=1)
        train_attention_mask = torch.cat((train_encodings_1['attention_mask'], train_encodings_2['attention_mask']), dim=1)
        test_input_ids = torch.cat((test_encodings_1['input_ids'], test_encodings_2['input_ids']), dim=1)
        test_attention_mask = torch.cat((test_encodings_1['attention_mask'], test_encodings_2['attention_mask']), dim=1)

        # Create TensorDatasets
        train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
        test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

        sts_train_iterator = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        sts_test_iterator = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        for epoch in range(config["epochs"]):
            print(f"Epoch {epoch}")
            model.train()
            total_loss = 0
            for i, batch in enumerate(sts_train_iterator):
                if i >= 5:
                    break
                print("Training")
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = mse_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                total_loss += loss.item()
            train_loss = total_loss / len(sts_train_iterator)
            print(f"Train loss: {train_loss}")
            # wandb.log({"train_loss": train_loss})

            model.eval()
            with torch.no_grad():
                total_loss = 0
                for i, batch in enumerate(sts_test_iterator):
                    if i >= 5:
                        break
                    print("Testing")
                    input_ids, attention_mask, labels = batch
                    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = mse_loss(outputs, labels)
                    total_loss += loss.item()
                test_loss = total_loss / len(sts_test_iterator)
                print(f"Test loss: {test_loss}")
                # wandb.log({"test_loss": test_loss})

    if config["task"] == "sentiment":
        model = AutoModelForSequenceClassification.from_pretrained(config["model_type"], num_labels=3)
        model.to(device)
        cls_dataset = load_dataset("csv", delimiter='\t', data_files={"train": "data-sent/csfd-train.tsv",
                                                                      "test": "data-sent/csfd-test.tsv"})
        cls_train_iterator = DataLoader(cls_dataset['train'], batch_size=config['batch_size'])
        cls_test_iterator = DataLoader(cls_dataset['test'], batch_size=config['batch_size'])

        cross_entropy = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        for epoch in range(config["epochs"]):
            print(f"Epoch {epoch}")
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            for i, batch in enumerate(cls_train_iterator):
                if i >= 5:
                    break
                print("Training")
                texts = batch["text"]
                labels = batch["label"].to(device)
                optimizer.zero_grad()
                encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=100).to(device)
                output = model(**encoded)
                loss = cross_entropy(output.logits, labels)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                predictions = torch.argmax(output.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += len(labels)
                total_loss += loss.item()
            train_acc = correct / total
            train_loss = total_loss / total
            print(f"Train loss: {train_loss}, Train acc: {train_acc}")
            # wandb.log({"train_loss": train_loss, "train_acc": train_acc})

            model.eval()
            with torch.no_grad():
                total_loss = 0
                correct = 0
                total = 0
                for i, batch in enumerate(cls_test_iterator):
                    if i >= 5:
                        break
                    print("Testing")
                    texts = batch["text"]
                    labels = batch["label"]
                    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=100).to(device)
                    output = model(**encoded)
                    predictions = torch.argmax(output.logits, dim=1)
                    loss = cross_entropy(output.logits, labels.to(device))
                    correct += (predictions == labels).sum().item()
                    total += len(labels)
                    total_loss += loss.item()
                val_acc = correct / total
                val_loss = total_loss / total
                print(f"Val acc: {val_acc}, val loss: {val_loss}")
                # wandb.log({"test_acc": val_acc, "train_loss": val_loss})


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--task", type=str, default="sts")
    argparser.add_argument("--model_type", type=str, default="UWB-AIR/Czert-B-base-cased")
    argparser.add_argument("--batch_size", type=int, default=16)
    argparser.add_argument("--lr", type=float, default=1e-5)
    argparser.add_argument("--epochs", type=int, default=10)
    config = argparser.parse_args()
    config = {
        # "task":"sts",     # < 0.65
        "task": "sentiment",  # > 0.75

        "model_type": "UWB-AIR/Czert-B-base-cased",
        # "model_type": "ufal/robeczech-base",
        # "model_type": "fav-kky/FERNET-C5",

        "batch_size": 16,
        "lr": 1e-5,
        "epochs": 10
    }

    main(config)
