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
        # self.model = BertForSequenceClassification.from_pretrained(model, num_labels=num_classes, output_attentions=False, output_hidden_states=False)
        self.model = model
        self.lr = learning_rate

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, label_batch = batch
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=label_batch)
        loss = outputs.loss

        metrics = {}
        metrics['train_loss'] = round(loss.item(), 2)

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred

    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.Tensor().to(device='cuda')
        true = torch.Tensor().to(device='cuda')
        pred = torch.Tensor().to(device='cuda')
        
        for output in validation_step_outputs:
            print(output[0])
            loss = torch.cat((loss, output[0]), dim=0)
            true = torch.cat((true, output[1]), dim=0)
            pred = torch.cat((pred, output[2]), dim=0)

        print(loss)
        sys.exit()

        # all_outputs = torch.stack(validation_step_outputs)

        # loss = torch.mean(all_outputs[:, 0]) 
        # true = all_outputs[:, 1]
        # pred = all_outputs[:, 2]

        cls_report = classification_report(true, pred, labels=[0, 1], output_dict=True, zero_division=0)

        accuracy = round(cls_report['accuracy'], 2)
        f1_score = round(cls_report['1']['f1-score'], 2)
        precision = round(cls_report['1']['precision'], 2)
        recall = round(cls_report['1']['recall'], 2)

        metrics = {}
        metrics['val_loss'] = round(loss.item(), 2)
        metrics['val_accuracy'] = accuracy
        metrics['val_f1_score'] = f1_score
        metrics['val_precision'] = precision
        metrics['val_recall'] = recall

        print(metrics)

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred

    def test_eopch_end(self, test_step_outputs):
        all_outputs = torch.stack(test_step_outputs)

        loss = torch.mean(all_outputs[:, 0]) 
        true = all_outputs[:, 1]
        pred = all_outputs[:, 2]

        cls_report = classification_report(true, pred, labels=[0, 1], output_dict=True, zero_division=0)

        metrics = {}
        metrics['test_loss'] = round(loss.item(), 2)
        metrics['test_accuracy'] = accuracy
        metrics['test_f1_score'] = f1_score
        metrics['test_precision'] = precision
        metrics['test_recall'] = recall

        print(metrics)

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

        return loss

    def _shared_eval_step(self, batch, batch_idx):
        input_ids, attention_mask, label_batch = batch
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=label_batch)
        loss = outputs.loss

        true = label_batch.to(torch.device("cpu"))
        pred = torch.argmax(F.softmax(outputs.logits, dim=1), dim=1).to(torch.device("cpu"))

        return loss, true, pred

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        outputs = self.model(input_ids, attention_mask)

        pred = torch.argmax(F.softmax(outputs.logits, dim=1), dim=1).to(torch.device("cpu"))

        return pred[0]
