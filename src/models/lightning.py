import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score


class lynModel(pl.LightningModule):
    def __init__(self, model, optimizer_lr=0.01, optimizer_momentum=0.9):
        super().__init__()
        self.model = model
        self.lr = optimizer_lr
        self.momentum = optimizer_momentum

    def training_step(self, batch, batch_idx):
        messages = batch["message"]
        labels = batch["label"]

        output = self.model(messages)
        loss = F.binary_cross_entropy(output, labels)

        self.log("train_loss", loss)

        preds = torch.zeros(output.shape)
        preds[output >= 0.5] = 1
        self.log("train_acc", torch.sum(preds == labels) / len(preds))

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
            on_epoch=True
        )
        self.log(
            "val_acc",
            (torch.sum(preds == labels) / len(labels)).numpy().sum(),
            on_epoch=True
        )

        return accuracy_score(labels.numpy(), preds.numpy())

    def validation_epoch_end(self, validation_step_outputs):
        ct, sum = 0, 0
        for pred in validation_step_outputs:
            sum += pred
            ct += 1
        return sum / ct

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum
        )
