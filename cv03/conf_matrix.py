import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

wandb_export = pd.read_csv("img/wandb_export_C.csv")


def plot_confusion_matrix(wandb_export):
    grouped = wandb_export.groupby(['Actual', 'Predicted']).sum().reset_index()
    conf_matrix = grouped.pivot(index='Actual', columns='Predicted', values='nPredictions')

    num_runs = wandb_export['Actual'].value_counts().sum() / 9
    conf_matrix = conf_matrix / num_runs

    plt.matshow(conf_matrix, cmap='Blues')
    plt.xlabel('Predicted')

    plt.ylabel('Actual')
    plt.title('Confusion matrix for the CNN C model')

    labels = ['neg', 'neu', 'pos']
    plt.xticks(ticks=[0, 1, 2], labels=labels)
    plt.yticks(ticks=[0, 1, 2], labels=labels)

    for (i, j), val in np.ndenumerate(conf_matrix):
        plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

    plt.savefig('img/conf_matrix_C.svg')
    plt.show()


plot_confusion_matrix(wandb_export)
