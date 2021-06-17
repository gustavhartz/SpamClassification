import torch

import torch.nn as nn
import torch.nn.functional as F
from transformers.file_utils import ModelOutput
import pytorch_lightning as pl

from sklearn.metrics import accuracy_score


class lynModel(pl.LightningModule):
    def __init__(self, model, optimizer_lr=0.01):
        super().__init__()
        self.model = model
        self.lr = optimizer_lr

    def training_step(self, batch, batch_idx):
        messages = batch["message"]
        labels = batch["label"]

        output = self.model(messages)
        loss = F.binary_cross_entropy(output, labels)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False
        )

        return loss

    def validation_step(self, batch, batch_idx):
        messages = batch["message"]
        labels = batch["label"]

        output = self.model(messages)
        loss = F.binary_cross_entropy(output, labels)

        # log validation loss per epoch
        preds = torch.zeros(output.shape)
        preds[output >= 0.5] = 1

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        return accuracy_score(labels.numpy(), preds.numpy())

    def validation_epoch_end(self, validation_step_outputs):
        ct, sum = 0, 0
        for pred in validation_step_outputs:
            sum += pred
            ct += 1

        # log sum/ct

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)