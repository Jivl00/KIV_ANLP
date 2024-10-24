# https://pytorch.org/data/main/tutorial.html
# https://towardsdatascience.com/text-classification-with-cnns-in-pytorch-1113df31e79f
import configparser
import os
import pickle
import sys
from collections import defaultdict

import random

import numpy as np
import torch
# from ignite.contrib.handlers.param_scheduler import create_lr_scheduler_with_warmup
from datasets import load_dataset

import wandb
from torch import nn
from torch.utils.data import DataLoader
from collections import Counter

import wandb_config
from cv02.consts import EMB_FILE
from cv02.main02 import dataset_vocab_analysis, MySentenceVectorizer, PAD, UNK

NUM_CLS = 3

CNN_MODEL = "cnn"
MEAN_MODEL = "mean"

CSFD_DATASET_TRAIN = "cv03/data/csfd-train.tsv"
CSFD_DATASET_TEST = "cv03/data/csfd-test.tsv"

CLS_NAMES = ["neg", "neu", "pos"]

from wandb_config import WANDB_PROJECT, WANDB_ENTITY

from cv02.main02 import dataset_vocab_analysis, load_ebs, MySentenceVectorizer, DummyModel, test, WORD2IDX, \
    VECS_BUFF
import sys


def count_statistics(dataset, vectorizer) -> tuple[float, dict]:
    ## todo CF#01
    for sentence in dataset["text"]:
        vectorizer.sent2idx(sentence)

    coverage = 1 - vectorizer.out_of_vocab_perc() / 100
    class_distribution = Counter(dataset["label"])
    # normalize
    for k in class_distribution:
        class_distribution[k] = class_distribution[k] / len(dataset["label"])

    return coverage, class_distribution


class MyBaseModel(torch.nn.Module):

    def __init__(self, config, w2v=None):
        super(MyBaseModel, self).__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.activation = None
        self.emb_layer = None
        self.emb_proj = None


class MyModelAveraging(MyBaseModel):
    def __init__(self, config, w2v=None):
        super(MyModelAveraging, self).__init__(config, w2v),

    def forward(self, x, l):
        return None


class MyModelConv(MyBaseModel):
    def __init__(self, config, w2v=None):
        super(MyModelConv, self).__init__(config, w2v),
        self.config = config

        # todo CF#CNN_CONF
        self.cnn_architecture = config["cnn_architecture"]  # for unit tests
        if self.cnn_architecture == "A":
            self.cnn_config = [(1, config["n_kernel"], (2, 1)), (1, config["n_kernel"], (3, 1)),
                               (1, config["n_kernel"], (4, 1))]
            self.config["hidden_size"] = 500
        elif self.cnn_architecture == "B":
            self.config["hidden_size"] = 514
            self.cnn_config = [(1, config["n_kernel"], (2, 2)), (1, config["n_kernel"], (3, 3)),
                               (1, config["n_kernel"], (4, 4))]
        elif self.cnn_architecture == "C":
            self.config["hidden_size"] = 35000
            self.cnn_config = [(1, config["n_kernel"], (2, config["proj_size"])),
                               (1, config["n_kernel"], (3, config["proj_size"])),
                               (1, config["n_kernel"], (4, config["proj_size"]))]

        # !!!!!! TODO
        # this line is important if you use list to group your architecture ... optimizer would not register if it is not used
        self.modules = nn.ModuleList(self.conv_layers)

    def forward(self, x, l):
        return None


def test_on_dataset(dataset_iterator, vectorizer, model, loss_metric_func):
    test_loss_list = []
    test_acc_list = []
    test_enum_y = []
    test_enum_pred = []

    for b in dataset_iterator:
        texts = b["text"]
        labels = b["label"]

        vectorized = [vectorizer.sent2idx(x) for x in texts]

        pred = model(vectorized)
        loss = loss_metric_func(pred, labels)
        test_loss_list.append(loss.item())

        pred_cls = torch.argmax(pred, dim=-1)
        acc = torch.sum(pred_cls == labels) / len(labels)
        test_acc_list.append(acc.item())

        test_enum_y.append(labels)
        test_enum_pred.append(pred_cls)


    return {
        "test_acc": sum(test_acc_list) / len(test_acc_list),
        "test_loss": sum(test_loss_list) / len(test_loss_list),
        "test_pred_clss": torch.cat(test_enum_pred),
        "test_enum_gold": torch.cat(test_enum_y),
    }


def load_ebs(emb_file, top_n_words: list, wanted_vocab_size, force_rebuild=False, random_emb=False):
    print("prepairing W2V...", end="")
    if random_emb:
        print("...random embeddings")
        word2idx = {}
        for i, w in enumerate(top_n_words[:wanted_vocab_size]):
            word2idx[w] = i
        word2idx[UNK] = len(word2idx)
        word2idx[PAD] = len(word2idx)
        vecs = np.random.uniform(-1, 1, (wanted_vocab_size + 2, 300))
        vecs[word2idx[PAD]] = np.zeros(300)
        assert len(vecs) == len(word2idx)

        return word2idx, vecs
    if os.path.exists(WORD2IDX) and os.path.exists(VECS_BUFF) and not force_rebuild:
        # CF#3
        print("...loading from buffer")
        with open(WORD2IDX, 'rb') as idx_fd, open(VECS_BUFF, 'rb') as vecs_fd:
            word2idx = pickle.load(idx_fd)
            vecs = pickle.load(vecs_fd)
    else:
        print("...creating from scratch")

        with open(emb_file, 'r', encoding="utf-8") as emb_fd:
            word2idx = {}
            vecs = []

            for idx, word in enumerate(top_n_words[:wanted_vocab_size]):
                word2idx[word] = idx

            # prune given word embeddings to the wanted_vocab_size
            vecs = np.random.uniform(-1, 1, (len(word2idx) + 2, 300))
            for i, l in enumerate(emb_fd):
                if i == 0:
                    continue
                l = l.strip().split(" ")
                word = l[0]
                if word in word2idx:
                    vecs[word2idx[word]] = np.array(l[1:], dtype=np.float32)

            word2idx[UNK] = len(word2idx)
            word2idx[PAD] = len(word2idx)
            vecs[word2idx[PAD]] = np.zeros(300)
            # vecs[word2idx[UNK]] = np.random.uniform(-1, 1, 300)

            # assert len(word2idx) > 6820
            assert len(vecs) == len(word2idx)
            pickle.dump(word2idx, open(WORD2IDX, 'wb'))
            pickle.dump(vecs, open(VECS_BUFF, 'wb'))

    return word2idx, vecs


def main(config: dict):
    cls_dataset = load_dataset("csv", delimiter='\t', data_files={"train": [CSFD_DATASET_TRAIN],
                                                                  "test": [CSFD_DATASET_TEST]})

    top_n_words = dataset_vocab_analysis(cls_dataset['train']["text"], top_n=-1)
    print("Top N words:", len(top_n_words))

    word2idx, word_vectors = load_ebs(EMB_FILE, top_n_words, config['vocab_size'], force_rebuild=False,
                                      random_emb=config["random_emb"])

    vectorizer = MySentenceVectorizer(word2idx, config["seq_len"])

    coverage, cls_dist = count_statistics(cls_dataset['train'], vectorizer)
    print(f"COVERAGE: {coverage}\ncls_dist:{cls_dist}")

    slit_dataset = cls_dataset['train'].train_test_split(test_size=0.2, shuffle=True)
    cls_dataset['train'] = slit_dataset['train']
    cls_dataset['valid'] = slit_dataset['test']

    cls_train_iterator = DataLoader(cls_dataset['train'], batch_size=config['batch_size'])
    cls_val_iterator = DataLoader(cls_dataset['valid'], batch_size=config['batch_size'])
    cls_test_iterator = DataLoader(cls_dataset['test'], batch_size=config['batch_size'])




    if config["model"] == CNN_MODEL:
        model = MyModelConv(config, w2v=word_vectors)
    elif config["model"] == MEAN_MODEL:
        model = MyModelAveraging(config, w2v=word_vectors)

    num_of_params = 0
    for x in model.parameters():
        print(x.shape)
        num_of_params += torch.prod(torch.tensor(x.shape), 0)
    config["num_of_params"] = num_of_params
    print("num of params:", num_of_params)

    # wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=["cv03"], config=config)
    # wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=["cv03","best"], config=config)

    model.to(config["device"])
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 0)

    batch = 0
    while True:
        for b in cls_train_iterator:
            texts = b["text"]
            labels = b["label"]

            vectorized = [vectorizer.sent2idx(x) for x in texts]

            optimizer.zero_grad()
            pred = model(vectorized)

            loss = cross_entropy(pred, labels)
            loss.backward()
            optimizer.step()


            if batch % 100 == 0:
                model.eval()
                ret = test_on_dataset(cls_val_iterator, vectorizer, model, cross_entropy)
                conf_matrix = wandb.plot.confusion_matrix(preds=ret["test_pred_clss"].cpu().numpy(),
                                                          y_true=ret["test_enum_gold"].cpu().numpy(),
                                                          class_names=CLS_NAMES)
                # wandb.log({"conf_mat":conf_matrix})

                # wandb.log({"test_acc": ret["test_acc"],
                #            "test_loss": ret["test_loss"]}, commit=False)
                model.train()

            # wandb.log(
            #     {"train_loss": loss, "train_acc": train_acc, "lr": lr_scheduler.get_last_lr()[0], "pred": pred,
            #      "norm": total_norm})
            batch += 1

        lr_scheduler.step()

        if batch >= config["batches"]:
            break

    ret = test_on_dataset(cls_test_iterator, vectorizer, model, cross_entropy)
    wandb.log({"final_test_acc": ret["test_acc"],
               "final_test_loss": ret["test_loss"]})




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using {device}")

    N_KERNEL = 64
    PROJ_SIZE = 100

    config = {"batch_size": 33,
              "vocab_size": 20000,
              # "model": MEAN_MODEL,
              "model": CNN_MODEL,
              "device": device,
              "n_kernel": N_KERNEL,

              "activation": "relu",
              "random_emb": False,
              "emb_training": False,
              "emb_projection": True,
              "proj_size": PROJ_SIZE,
              "cnn_architecture": "C",

              "emb_size": 300,
              "lr": 0.0005,
              "gradient_clip": .5,
              'batches': 200000,
              "seq_len": 100
              }

    main(config)
