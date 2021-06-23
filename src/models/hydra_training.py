import os

import hydra
import pytorch_lightning as pl
import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from src.data.data_utils import SPAMorHAMDataset
from src.models.lightning import lynModel
from src.models.model import LSTM
import os

load_dotenv(find_dotenv())


@hydra.main(config_path="../../config", config_name="config_run_1.yaml")
def my_app(cfg: DictConfig) -> None:
    # Hyperparameters
    token_len = cfg.data.token_len  # 40
    batch_size = cfg.data.batch_size  # 128
    hidden_dim = cfg.model.hidden_dim  # 128
    embedding_dim = cfg.model.embedding_dim  # 20
    dropout = cfg.model.dropout  # 0.5
    lr = cfg.model.lr  # 0.01
    epochs = cfg.model.epochs  # 20
    seed = cfg.enviroment.torch_seed

    torch.manual_seed(seed)

    # For shuffle dataloader
    g = torch.Generator()
    g.manual_seed(seed)

    print(os.getcwd())

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    OG_path = hydra.utils.get_original_cwd()
    train_set = SPAMorHAMDataset(
        os.path.join(OG_path, "data/processed/test_set.csv"),
        input_dim=token_len,
        tokenizer=tokenizer,
    )
    val_set = SPAMorHAMDataset(
        os.path.join(OG_path, "data/processed/val_set.csv"),
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

    litmodel = lynModel(model, optimizer_lr=lr)
    if cfg.enviroment.wandb:
        hyper = {}
        hyper.update(cfg["model"])
        hyper.update(cfg["data"])
        wandb_logger = WandbLogger(project="MLOPS_SpamHam", config=hyper)
        trainer = pl.Trainer(
            max_epochs=epochs, logger=wandb_logger, log_every_n_steps=10
        )
        wandb_logger.log_hyperparams(hyper)
    else:
        trainer = pl.Trainer(max_epochs=epochs)

    trainer.fit(litmodel, trainloader, validloader)

    # save model in hydra folder
    torch.save(litmodel.model, "model.model")


if __name__ == "__main__":
    my_app()
