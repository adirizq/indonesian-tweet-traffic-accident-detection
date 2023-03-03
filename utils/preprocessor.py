import os
import re
import sys
import torch
import emoji
import string
import multiprocessing
import pytorch_lightning as pl
import pandas as pd

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


class TwitterDataModule(pl.LightningDataModule):

    def __init__(self, tokenizer, max_length=128, batch_size=32) -> None:

        super(TwitterDataModule, self).__init__()

        self.seed = 42
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.dataset_path = "datasets/twitter_label_manual.csv"
        self.processed_dataset_path = "datasets/twitter_label_manual_processed.csv"

    def load_data(self):
        if os.path.exists(self.processed_dataset_path):
            print('[ Loading Dataset ]')
            dataset = pd.read_csv(self.processed_dataset_path)
            print('[ Load Completed ]\n')
        else:
            print('[ Preprocessing Dataset ]')
            dataset = pd.read_csv(self.dataset_path)[["full_text", "is_accident"]]
            dataset.columns = ["text", "label"]
            dataset.dropna(subset=['text'], inplace=True)

            self.stop_words = StopWordRemoverFactory().get_stop_words()

            tqdm.pandas(desc='Preprocessing')
            dataset["text"] = dataset["text"].progress_apply(lambda x: self.clean_tweet(x))
            dataset.dropna(subset=['text'], inplace=True)
            print('[ Preprocess Completed ]\n')

            print('[ Saving Preprocessed Dataset ]')
            dataset.to_csv(self.processed_dataset_path, index=False)
            print('[ Save Completed ]\n')

        total_size = len(dataset.index)

        print('[ Tokenizing Dataset ]')
        x_input_ids, x_attention_mask, y = [], [], []
        for (text, label) in tqdm(dataset.values.tolist()):

            encoded_label = [0, 0]
            encoded_label[label] = 1

            encoded_text = self.tokenizer(text=text,
                                          max_length=self.max_length,
                                          padding="max_length",
                                          truncation=True)

            x_input_ids.append(encoded_text['input_ids'])
            x_attention_mask.append(encoded_text['attention_mask'])
            y.append(encoded_label)

        x_input_ids = torch.tensor(x_input_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        y = torch.tensor(y)

        del (dataset)

        tensor_dataset = TensorDataset(x_input_ids, x_attention_mask, y)
        print('[ Tokenize Completed ]\n')

        print('[ Splitting Dataset ]')
        train_validation_size = int(0.8 * total_size)
        train_size = int(0.9 * train_validation_size)
        validation_size = train_validation_size - train_size
        test_size = total_size - train_validation_size

        train_validation_dataset, test_dataset = torch.utils.data.random_split(tensor_dataset, [train_validation_size, test_size], generator=torch.Generator().manual_seed(42))
        train_dataset, validation_dataset = torch.utils.data.random_split(train_validation_dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(42))
        print('[ Split Completed ]\n')

        return train_dataset, validation_dataset, test_dataset

    def clean_tweet(self, text):
        result = text.lower()
        result = self.remove_emoji(result)  # remove emoji
        result = re.sub(r'\n', ' ', result)  # remove new line
        result = re.sub(r'@\w+', '', result)  # remove user mention
        result = re.sub(r'http\S+', '', result)  # remove link
        result = re.sub(r'\d+', '', result)  # remove number
        result = re.sub(r'[^a-zA-Z ]', '', result)  # get only alphabets
        result = ' '.join([word for word in result.split() if word not in self.stop_words])  # remove stopword
        result = result.strip()

        if result == '':
            result = float('NaN')

        return result

    def remove_emoji(self, text):
        return emoji.replace_emoji(text, replace='')

    def setup(self, stage=None):
        train_data, valid_data, test_data = self.load_data()
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "test":
            self.test_data = test_data

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count()
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )
