import wandb
import sys
import torch
import os

import numpy as np

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models.model import LSTM
from src.data.data_utils import SPAMorHAMDataset
from src.models.lightning import lynModel

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../", config_name="config")
def my_app(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    # Hyperparameters
    token_len = cfg.token_len  # 40
    batch_size = cfg.batch_size  # 128
    hidden_dim = cfg.hidden_dim  # 128
    embedding_dim = cfg.embedding_dim  # 20
    dropout = cfg.dropout  # 0.5
    lr = cfg.lr  # 0.01
    epochs = cfg.epochs  # 20

    seed = cfg.seed

    torch.manual_seed(seed)

    # For shuffle dataloader
    g = torch.Generator()
    g.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    OG_path = hydra.utils.get_original_cwd()
    train_set = SPAMorHAMDataset(
        os.path.join(OG_path, "./data/processed/test_set.csv"),
        input_dim=token_len,
        tokenizer=tokenizer,
    )
    val_set = SPAMorHAMDataset(
        os.path.join(OG_path, "./data/processed/val_set.csv"),
        input_dim=token_len,
        tokenizer=tokenizer,
    )

    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    validloader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )

    # Model
    model = LSTM(
        tokenizer.vocab_size,
        hidden_dimension=hidden_dim,
        embedding_dim=embedding_dim,
        text_len=token_len,
        dropout=dropout,
    )

    trainer = pl.Trainer(max_epochs=epochs)
    litmodel = lynModel(model, optimizer_lr=lr)

    trainer.fit(litmodel, trainloader, validloader)


if __name__ == "__main__":
    my_app()
