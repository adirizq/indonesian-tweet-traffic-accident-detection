import sys
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from transformers import BertForSequenceClassification
from sklearn.metrics import classification_report


class Finetune(pl.LightningModule):

    def __init__(self, model, learning_rate=2e-5) -> None:

        super(Finetune, self).__init__()
        self.model = model
        self.lr = learning_rate

        self.linear1 = nn.Linear(768, 32)
        self.linear2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask):
        bert_output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        linear_output = self.linear1(bert_output.pooler_output)
        relu_output = self.relu(linear_output)

        dropout_output = self.dropout(relu_output)

        linear_output = self.linear2(dropout_output)
        sigmoid_output = self.sigmoid(linear_output)

        return sigmoid_output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask))

        loss = self.criterion(outputs, targets)

        metrics = {}
        metrics['train_loss'] = loss.item()

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred

    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.Tensor().to(device='cuda')
        true = []
        pred = []

        for output in validation_step_outputs:
            loss = torch.cat((loss, output[0].view(1)), dim=0)
            true += output[1].numpy().tolist()
            pred += output[2].numpy().tolist()

        loss = torch.mean(loss)

        cls_report = classification_report(true, pred, labels=[0, 1], output_dict=True, zero_division=0)

        accuracy = cls_report['accuracy']
        f1_score = cls_report['1']['f1-score']
        precision = cls_report['1']['precision']
        recall = cls_report['1']['recall']

        metrics = {}
        metrics['val_loss'] = loss.item()
        metrics['val_accuracy'] = accuracy
        metrics['val_f1_score'] = f1_score
        metrics['val_precision'] = precision
        metrics['val_recall'] = recall

        print()
        print(metrics)

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred

    def test_epoch_end(self, test_step_outputs):
        loss = torch.Tensor().to(device='cuda')
        true = []
        pred = []

        for output in test_step_outputs:
            loss = torch.cat((loss, output[0].view(1)), dim=0)
            true += output[1].numpy().tolist()
            pred += output[2].numpy().tolist()

        loss = torch.mean(loss)

        cls_report = classification_report(true, pred, labels=[0, 1], output_dict=True, zero_division=0)

        accuracy = cls_report['accuracy']
        f1_score = cls_report['1']['f1-score']
        precision = cls_report['1']['precision']
        recall = cls_report['1']['recall']

        metrics = {}
        metrics['test_loss'] = loss.item()
        metrics['test_accuracy'] = accuracy
        metrics['test_f1_score'] = f1_score
        metrics['test_precision'] = precision
        metrics['test_recall'] = recall

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

        return loss

    def _shared_eval_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)

        print(len(outputs.size()))

        if len(outputs.size()) >= 2:
            outputs = torch.squeeze(outputs)

        print(outputs.size())
        print(targets.size())

        loss = self.criterion(outputs, targets)

        true = targets.to(torch.device("cpu"))
        pred = (outputs >= 0.5).int().to(torch.device("cpu"))

        return loss, true, pred

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask))

        pred = (outputs >= 0.5).int().to(torch.device("cpu"))

        return pred[0]
