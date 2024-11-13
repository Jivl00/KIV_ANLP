import os

import transformers
import matplotlib.pyplot as plt


# Load data
def load_data(file_name, folder):
    file_path = os.path.join(folder, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = []
        sentence = {"words": [], "labels": []}
        for line in file:
            if line == "\n":
                if len(sentence["words"]) > 0:
                    sentences.append(sentence)
                    sentence = {"words": [], "labels": []}
            else:
                splits = line.strip().split(' ')
                word = splits[0]
                if len(splits) > 1:
                    tag = splits[-1]
                else:
                    tag = 'O'
                sentence["words"].append(word)
                sentence["labels"].append(tag)

        if len(sentence["words"]) > 0:
            sentences.append(sentence)

    return sentences

def avg_sentence_length(data):
    total = 0
    for sentence in data:
        total += len(sentence["words"])
    return total / len(data)

def avg_token_length(data):
    tokenizer = transformers.BertTokenizerFast.from_pretrained("UWB-AIR/Czert-B-base-cased")
    total = 0
    for sentence in data:
        sentence_text = " ".join(sentence["words"])
        tokens = tokenizer.tokenize(sentence_text)
        total += len(tokens)
    return total / len(data)

def main():
    data_folder_CNEC = 'data'
    data_folder_UD = 'data-mt'
    files = ['train.txt', 'test.txt', 'dev.txt']

    print("CNEC")
    overall = 0
    for file in files:
        data = load_data(file, data_folder_CNEC)
        print(f"{file}: {len(data)}")
        overall += len(data)

    print(f"Overall: {overall}")


    print("=====")
    print("UD")
    overall = 0
    for file in files:
        data = load_data(file, data_folder_UD)
        print(f"{file}: {len(data)}")
        overall += len(data)

    print(f"Overall: {overall}")

    print("=====")
    print("Average sentence length")
    print("CNEC")
    for file in files:
        data = load_data(file, data_folder_CNEC)
        print(f"{file}: {avg_sentence_length(data)}")
    print("=====")
    print("UD")
    for file in files:
        data = load_data(file, data_folder_UD)
        print(f"{file}: {avg_sentence_length(data)}")

    print("=====")
    print("Average token length")
    print("CNEC")
    for file in files:
        data = load_data(file, data_folder_CNEC)
        print(f"{file}: {avg_token_length(data)}")
    print("=====")
    print("UD")
    for file in files:
        data = load_data(file, data_folder_UD)
        print(f"{file}: {avg_token_length(data)}")

    print("=====")
    print("Statistics about class distribution")
    print("CNEC")
    for file in files:
        data = load_data(file, data_folder_CNEC)
        classes = {}
        for sentence in data:
            for label in sentence["labels"]:
                if label not in classes:
                    classes[label] = 0
                classes[label] += 1
        # histogram
        plt.bar(classes.keys(), classes.values(), color='skyblue')
        plt.title(f"{file} CNEC class distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.grid()
        plt.savefig(f"img/{file}_CNEC_class_distribution.svg")
        plt.show()

    print("=====")
    print("UD")
    for file in files:
        data = load_data(file, data_folder_UD)
        classes = {}
        for sentence in data:
            for label in sentence["labels"]:
                if label not in classes:
                    classes[label] = 0
                classes[label] += 1
        # histogram
        # fig = plt.figure(figsize=(10, 8))
        plt.bar(classes.keys(), classes.values(), color='skyblue')
        plt.title(f"{file} UD class distribution")
        plt.xticks(rotation=45)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.grid()
        plt.savefig(f"img/{file}_UD_class_distribution.svg")
        plt.show()

    print("=====")



if __name__ == '__main__':
    main()
