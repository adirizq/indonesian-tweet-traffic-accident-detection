import pandas as pd
import matplotlib.pyplot as plt
import os

from tqdm import tqdm


if __name__ == '__main__':
    # model_name = ['IndoBERT', 'IndoBERTweet', 'RoBERTa OSCAR', 'RoBERTa Wiki']
    # batch = ['16', '32']
    # bert_lr = ['2e-05', '3e-05', '5e-05']
    # roberta_lr = ['1e-05', '2e-05', '3e-05']
    # metric = [
    #     ['val_loss', 'Validation Loss', 'Loss'],
    #     ['val_accuracy', 'Validation Accuracy', 'Accuracy'],
    #     ['val_f1', 'Validation F1-Score', 'F1-Score'],
    #     ['loss', 'Train Loss', 'Loss'],
    #     ['accuracy', 'Train Accuracy', 'Accuracy'],
    #     ['f1', 'Train F1-Score', 'F1-Score'],
    # ]

    # path = 'results/history'

    # for model in tqdm(model_name, desc='Generating graph'):
    #     for m in metric:
    #         for b in batch:
    #             if model == 'IndoBERT' or model == 'IndoBERTweet':
    #                 for lr in bert_lr:
    #                     data = pd.read_csv(f'{path}/{model}_batch={b}_lr={lr}.csv')
    #                     data.rename(columns={data.columns[0]: "epoch"}, inplace=True)
    #                     data.epoch += 1

    #                     plt.plot(data.epoch, data[m[0]], label=f'batch={b}, lr={lr}')

    #             if model == 'RoBERTa OSCAR' or model == 'RoBERTa Wiki':
    #                 for lr in roberta_lr:
    #                     data = pd.read_csv(f'{path}/{model}_batch={b}_lr={lr}.csv')
    #                     data.rename(columns={data.columns[0]: "epoch"}, inplace=True)
    #                     data.epoch += 1

    #                     plt.plot(data.epoch, data[m[0]], label=f'batch={b}, lr={lr}')

    #         plt.title(f'{model} {m[1]}')
    #         plt.xlabel('Epoch')
    #         plt.ylabel(m[2])
    #         plt.legend()

    #         save_path = f'results/graph/{model}'
    #         if not os.path.exists(save_path):
    #             os.makedirs(save_path)

    #         plt.savefig(f'{save_path}/{m[1]}.png', facecolor='white', dpi=300)
    #         plt.close()

    best_model = [
        ['IndoBERT', '32', '5e-05'],
        ['IndoBERTweet', '32', '2e-05'],
        ['RoBERTa OSCAR', '32', '1e-05'],
        ['RoBERTa Wiki', '32', '1e-05']
    ]
    metric = [
        ['val_loss', 'Validation Loss', 'Loss'],
        ['val_accuracy', 'Validation Accuracy', 'Accuracy'],
        ['val_f1', 'Validation F1-Score', 'F1-Score'],
        ['loss', 'Train Loss', 'Loss'],
        ['accuracy', 'Train Accuracy', 'Accuracy'],
        ['f1', 'Train F1-Score', 'F1-Score'],
    ]

    path = 'results/history'

    for m in tqdm(metric, desc='Generating graph'):
        for model in best_model:
            data = pd.read_csv(f'{path}/{model[0]}_batch={model[1]}_lr={model[2]}.csv')
            data.rename(columns={data.columns[0]: "epoch"}, inplace=True)
            data.epoch += 1

            plt.plot(data.epoch, data[m[0]], label=f'{model[0]}')

            plt.title(f'All Best Models {m[1]}')
            plt.xlabel('Epoch')
            plt.ylabel(m[2])
            plt.legend()

        save_path = f'results/graph/all models'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.savefig(f'{save_path}/{m[1]}.png', facecolor='white', dpi=300)
        plt.close()
