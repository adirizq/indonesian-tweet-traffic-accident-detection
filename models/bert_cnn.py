import sys
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from transformers import BertForSequenceClassification
from sklearn.metrics import classification_report


class BERTFamilyCNN(pl.LightningModule):

    def __init__(self,
                 model,
                 learning_rate=2e-5,
                 in_channels=4,
                 out_channels=256,
                 window_sizes=[3, 4, 5],
                 embedding_size=768,) -> None:

        super(BERTFamilyCNN, self).__init__()
        self.model = model
        self.lr = learning_rate
        self.dropout_prob = 0.5
        self.relu_dim_list = [1024, 256, 128, 64, 32]

        self.conv2d = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, [window_size, embedding_size], padding=(window_size - 1, 0)) for window_size in window_sizes
        ])

        # FC classifier
        self.hidden2dense = nn.Linear(out_channels * len(window_sizes) + 768, self.relu_dim_list[0])

        modules = []
        for i in range(len(self.relu_dim_list) - 1):
            modules.append(nn.Linear(self.relu_dim_list[i], self.relu_dim_list[i + 1]))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(self.dropout_prob))
        dense2label = nn.Linear(self.relu_dim_list[-1], 1)
        modules.append(dense2label)

        self.classifier = nn.Sequential(*modules)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.sigmoid = nn.Sigmoid()

        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask):
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        pooler_output = model_output.pooler_output
        hs_output = torch.stack(model_output.hidden_states, dim=1)[:, -4:, 1:]

        out_conv = []

        for conv in self.conv2d:
            x = conv(hs_output)
            x = self.relu(x)
            x = x.squeeze(-1)
            x = F.max_pool1d(x, x.size(2))
            out_conv.append(x)

        logits = torch.cat(out_conv, 1).squeeze(-1)
        pooler_cnn_out = torch.cat([logits, pooler_output], 1)

        # Pass the output to
        dropout_out = self.dropout(pooler_cnn_out)
        dense = self.hidden2dense(dropout_out)
        relu_out = self.relu(dense)
        dropout_out = self.dropout(relu_out)

        classifier_out = self.classifier(dropout_out)

        return self.sigmoid(classifier_out)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask), dim=1)

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
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask), dim=1)

        loss = self.criterion(outputs, targets)

        true = targets.to(torch.device("cpu"))
        pred = (outputs >= 0.5).int().to(torch.device("cpu"))

        return loss, true, pred

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask), dim=1)

        pred = (outputs >= 0.5).int().to(torch.device("cpu"))

        return pred[0]
