import json
import os

from cv02.consts import EMB_FILE, TRAIN_DATA, TEST_DATA

cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

import argparse
import datetime

import pickle
import random
import sys
from collections import Counter

import wandb
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ExponentialLR

from matplotlib import pyplot as plt

import numpy as np
import torch as torch

with open('wandb_config.json', encoding="utf-8") as f:
    wandb_config = json.load(f)
print(f"loaded wandb_config: {wandb_config}")

WORD2IDX = "word2idx.pckl"
VECS_BUFF = "vecs.pckl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

DETERMINISTIC_SEED = 10
random.seed(DETERMINISTIC_SEED)

UNK = "<UNK>"
PAD = "<PAD>"

MAX_SEQ_LEN = 50

# BATCH_SIZE = 1000
MINIBATCH_SIZE = 10

EPOCH = 7

run_id = random.randint(100_000, 999_999)


#  file_path is path to the source file for making statistics
#  top_n integer : how many most frequent words
def dataset_vocab_analysis(texts, top_n=-1):
    counter = Counter()
    #  CF#1
    counter.update([word for l in texts for word in l.split(" ")])
    if top_n > 0:
        return list(dict(counter.most_common(top_n)).keys())
    return list(dict(counter.most_common()).keys())


#  emb_file : a source file with the word vectors
#  top_n_words : enumeration of top_n_words for filtering the whole word vector file
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
            idx = 0
            word2idx = {}
            vecs = []

            # CF#2
            #  create map of  word->id  of top according to the given top_n_words
            #  create a matrix as a np.array : word vectors
            #  vocabulary ids corresponds to vectors in the matrix
            #  Do not forget to add UNK and PAD tokens into the vocabulary.

            top_n_words = set(top_n_words[:wanted_vocab_size])

            # prune given word embeddings to the wanted_vocab_size
            vecs = np.zeros((len(top_n_words) + 2, 300))
            idx = 0
            for i, l in enumerate(emb_fd):
                if i == 0:
                    continue
                l = l.strip().split(" ")
                word = l[0]
                # if word in top_n_words:
                if word in top_n_words:
                    word2idx[word] = idx
                    vecs[idx] = np.array(l[1:], dtype=np.float32)
                    idx += 1

            # remove unused rows
            vecs = vecs[:idx + 2]  # +2 for UNK and PAD
            word2idx[UNK] = len(word2idx)
            word2idx[PAD] = len(word2idx)
            vecs[word2idx[PAD]] = np.zeros(300)
            vecs[word2idx[UNK]] = np.random.uniform(-1, 1, 300)

            assert len(word2idx) > 6820
            assert len(vecs) == len(word2idx)
            pickle.dump(word2idx, open(WORD2IDX, 'wb'))
            pickle.dump(vecs, open(VECS_BUFF, 'wb'))

    return word2idx, vecs


# This class is used for transforming text into sequence of ids corresponding to word vectors (using dict word2idx).
# It also counts some usable statistics.
class MySentenceVectorizer():
    def __init__(self, word2idx, max_seq_len):
        self._all_words = 0
        self._out_of_vocab = 0
        self.word2idx = word2idx
        self.max_seq_len = max_seq_len

    def sent2idx(self, sentence):
        idx = []
        # CF#4
        #  Transform sentence into sequence of ids using self.word2idx
        #  Keep the counters self._all_words and self._out_of_vocab up to date
        #  for checking coverage -- it is also used for testing.
        words = sentence.split(" ")[0:self.max_seq_len]
        for word in words:
            self._all_words += 1
            if word not in self.word2idx:
                self._out_of_vocab += 1
                idx.append(self.word2idx[UNK])
            else:
                idx.append(self.word2idx[word])

        return idx + [self.word2idx[PAD]] * (self.max_seq_len - len(idx))

    def out_of_vocab_perc(self):
        return (self._out_of_vocab / self._all_words) * 100

    def reset_counter(self):
        self._out_of_vocab = 0
        self._all_words = 0


# Load and preprocess the data from file.
class DataLoader():
    # vectorizer : MySentenceVectorizer
    def __init__(self, vectorizer, data_file_path, batch_size):
        self._data_folder = data_file_path
        self._batch_size = batch_size
        self.a = []
        self.b = []
        self.sts = []
        self.pointer = 0
        self._vectorizer = vectorizer
        print(f"loading data from {self._data_folder} ...")
        self.__load_from_file(self._data_folder)

        self.out_of_vocab = self._vectorizer.out_of_vocab_perc()
        self._vectorizer.reset_counter()

    def __load_from_file(self, file):
        #  CF#5
        #  load and preprocess the data set from file into self.a self.b self.sts
        #  use vectorizer to store only ids instead of strings
        with open(file, 'r', encoding="utf-8") as fd:
            for i, l in enumerate(fd):
                fragments = l.strip().split("\t")
                self.a.append(self._vectorizer.sent2idx(fragments[0]))
                self.b.append(self._vectorizer.sent2idx(fragments[1]))
                self.sts.append(float(fragments[2]))

                # You can use this snippet for faster debuging
                # if i == 4000:
                #     break

    def __iter__(self):
        #   CF#7
        #   randomly shuffle data in memory and start from begining
        self.pointer = 0
        zipped = list(zip(self.a, self.b, self.sts))
        random.shuffle(zipped)
        self.a, self.b, self.sts = zip(*zipped)

        self.pointer = 0

        return self

    def __next__(self):
        #   CF#6
        #   Implement yielding a batches from preloaded data: self.a,  self.b, self.sts
        batch = dict()
        if self.pointer + self._batch_size > len(self.a):
            raise StopIteration
        batch['a'] = np.array(self.a[self.pointer:self.pointer + self._batch_size])
        batch['b'] = np.array(self.b[self.pointer:self.pointer + self._batch_size])
        batch['sts'] = np.array(self.sts[self.pointer:self.pointer + self._batch_size])
        self.pointer += self._batch_size

        return batch


class TwoTowerModel(torch.nn.Module):
    def __init__(self, vecs, config):
        super(TwoTowerModel, self).__init__()
        #   CF#10a
        #   Initialize building block for architecture described in the assignment
        self.final_metric = config["final_metric"]
        self.config = config

        # torch.nn.Embedding
        # torch.nn.Linear

        self.emb_layer = torch.nn.Embedding.from_pretrained(torch.tensor(vecs), freeze=not self.config["emb_training"])

        self.emb_proj = torch.nn.Linear(300, 128)

        print("requires grads? : ", self.emb_layer.weight.requires_grad)
        self.relu = torch.nn.ReLU()

        # self.final_proj_1 = torch.nn.Linear(256, 128)
        if not config["emb_projection"]:
            self.final_proj_1 = torch.nn.Linear(600, 128)
        else:
            self.final_proj_1 = torch.nn.Linear(256, 128)

        self.final_proj_2 = torch.nn.Linear(128, 1)

    def _make_repre(self, idx):
        # embedding - > projection -> avg
        emb = self.emb_layer(idx)
        if self.config["emb_projection"]:
            proj = self.emb_proj(emb.float())
            proj = self.relu(proj)
        else:
            proj = emb.float()
        # proj = self.emb_proj(emb.float())
        avg = torch.mean(proj, 1)
        return avg

    def forward(self, batch):
        repre_a = self._make_repre(torch.tensor(batch['a']).to(device))
        repre_b = self._make_repre(torch.tensor(batch['b']).to(device))

        #    CF#10b
        #   Implement forward pass for the model architecture described in the assignment.
        #   Use both described similarity measures.
        if self.final_metric == "neural":
            repre = torch.cat((repre_a, repre_b), 1)
            repre = self.final_proj_1(repre.float())
            repre = self.relu(repre)
            repre = self.final_proj_2(repre)
            repre = torch.squeeze(repre)

        if self.final_metric == "cos":
            repre = torch.nn.functional.cosine_similarity(repre_a, repre_b, dim=1)

        return repre


class DummyModel(torch.nn.Module):  # predat dataset a vracet priod
    def __init__(self, train_loader):
        super(DummyModel, self).__init__()
        # for i, td in enumerate(train_loader):
        #     self.mean_on_train = torch.mean(td['sts']).item()
        #     break

        #   CF#9
        #   Implement DummyModel as described in the assignment.
        self.mean_on_train = torch.mean(torch.tensor(train_loader.sts)).item()
        print(f"mean_on_train: {self.mean_on_train}")

    def forward(self, batch):
        return torch.tensor([self.mean_on_train for _ in range(len(batch['a']))]).to(device)


# CF#8b
# process whole dataset and return loss
# save loss from each batch and divide it by all on the end.
def test(data_set, net, loss_function):
    running_loss = 0
    all = 0
    with torch.no_grad():
        for i, td in enumerate(data_set):
            predicted_sts = net(td)
            real_sts = torch.tensor(td['sts']).to(device)
            loss = loss_function(real_sts, predicted_sts)
            running_loss += loss.item()
            all += 1

    test_loss = running_loss / all
    print(f"test_loss:{test_loss}")
    return test_loss


def train_model(train_dataset, test_dataset, w2v, loss_function, config):
    # net = CzertModel()
    # net = net.to(device)
    net = TwoTowerModel(w2v, config)
    net = net.to(device)
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=config["lr"])
    if config["lr_scheduler"] == "step":
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    else:
        lr_scheduler = ExponentialLR(optimizer, gamma=0.1)

    train_loss_arr = []
    test_loss_arr = []

    running_loss = 0.0
    sample = 0
    x_axes = []  # vis

    # CF#8a Implement training loop
    for epoch in range(EPOCH):
        for i, td in enumerate(train_dataset):
            batch = td
            real_sts = torch.tensor(batch['sts']).to(device)

            optimizer.zero_grad()
            predicted_sts = net(batch)

            # if         "final_metric": "cos",
            #         "emb_training": False,
            #         "emb_projection": False,
            # don't backpropagate - there is no trainable parameters for this case
            if config["final_metric"] == "cos" and not config["emb_training"] and not config["emb_projection"]:
                loss = loss_function(real_sts.float(), predicted_sts.float())
            else:
                loss = loss_function(real_sts.float(), predicted_sts.float())
                loss.backward()
            optimizer.step()

            running_loss += loss.item()
            sample += BATCH_SIZE
            wandb.log({"train_loss": loss, "lr": lr_scheduler.get_last_lr()}, commit=False)

            if i % MINIBATCH_SIZE == MINIBATCH_SIZE - 1:
                train_loss = running_loss / MINIBATCH_SIZE
                running_loss = 0.0

                train_loss_arr.append(train_loss)

                net.eval()
                test_loss = test(test_dataset, net, loss_function)
                net.train()
                test_loss_arr.append(test_loss)

                wandb.log({"test_loss": test_loss}, commit=False)

                print(f"e{epoch} b{i}\ttrain_loss:{train_loss}\ttest_loss:{test_loss}\tlr:{lr_scheduler.get_last_lr()}")
            wandb.log({})

        lr_scheduler.step()

    print('Finished Training')
    os.makedirs("log", exist_ok=True)
    timestring = datetime.datetime.now().strftime("%b-%d-%Y--%H-%M-%S")
    plt.savefig(f"log/{run_id}-{timestring}.pdf")

    return test_loss_arr


def main(config=None):
    print("config:", config)
    global BATCH_SIZE
    BATCH_SIZE = config["batch_size"]
    config_str = json.dumps(config)
    # wandb.init(project=wandb_config["WANDB_PROJECT"], entity=wandb_config["WANDB_ENTITY"], tags=["cv02"], config=config,
    #            name=config_str)

    with open(TRAIN_DATA, 'r', encoding="utf-8") as fd:
        train_data_texts = fd.read().split("\n")

    top_n_words = dataset_vocab_analysis(train_data_texts, -1)
    print(len(top_n_words))

    word2idx, word_vectors = load_ebs(EMB_FILE, top_n_words, config['vocab_size'], random_emb=config["random_emb"])

    vectorizer = MySentenceVectorizer(word2idx, MAX_SEQ_LEN)
    # EXPECTED = [259, 642, 249, 66, 252, 3226]
    # print("EXPECTED:", EXPECTED)
    # sentence = "Podle vlády dnes není dalších otázek"
    # print(vectorizer.sent2idx(sentence))

    train_dataset = DataLoader(vectorizer, TRAIN_DATA, BATCH_SIZE)
    test_dataset = DataLoader(vectorizer, TEST_DATA, BATCH_SIZE)

    # dummy_net = DummyModel(train_dataset)
    # dummy_net = dummy_net.to(device)
    #
    loss_function = torch.nn.MSELoss()
    #
    # tests = [test(test_dataset, dummy_net, loss_function) for i in range(12)]
    # print("mean test loss:", np.mean(tests), "+-", np.std(tests))

    # test(train_dataset, dummy_net, loss_function)
    # train_model(train_dataset, test_dataset, word_vectors, loss_function, config)


if __name__ == '__main__':
    my_config = {
        "vocab_size": 20000,
        "random_emb": True
    }

    print(my_config)
    main(my_config)
