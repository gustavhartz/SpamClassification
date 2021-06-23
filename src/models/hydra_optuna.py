import os
import hydra
import pytorch_lightning as pl
import torch
import optuna

from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from src.data.data_utils import SPAMorHAMDataset
from src.models.lightning import lynModel
from src.models.model import LSTM

load_dotenv(find_dotenv())

@hydra.main(config_path="../../config", config_name="config_hydra_optuna.yaml")
def my_app(cfg: DictConfig) -> None:
    print('start!')
    # Hyperparameters
    epochs = cfg.model.epochs  # 20
    seed = cfg.environment.torch_seed

    torch.manual_seed(seed)

    # For shuffle dataloader
    g = torch.Generator()
    g.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    OG_path = hydra.utils.get_original_cwd()

    def optuna_objective(trial):
        batch_size = cfg.data.batch_size  # 128
        token_len = int(trial.suggest_discrete_uniform('token_len',cfg.data.token_len.low,cfg.data.token_len.high,cfg.data.token_len.step))
        hidden_dim = int(trial.suggest_discrete_uniform('hidden_dim', cfg.model.hidden_dim.low, cfg.model.hidden_dim.high, cfg.model.hidden_dim.step))
        embedding_dim = int(trial.suggest_discrete_uniform('embedding_dim', cfg.model.embedding_dim.low, cfg.model.embedding_dim.high, cfg.model.embedding_dim.step))
        dropout = trial.suggest_uniform('dropout',cfg.model.dropout.low,cfg.model.dropout.high)
        lr = trial.suggest_loguniform('lr', cfg.model.lr.low, cfg.model.lr.high)
        
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

        model = LSTM(
            tokenizer.vocab_size,
            hidden_dimension=hidden_dim,
            embedding_dim=embedding_dim,
            text_len=token_len,
            dropout=dropout,
        )

        if cfg.environment.wandb:
            hyper = {}
            hyper.update(cfg['model'])
            hyper.update(cfg['data'])
            wandb_logger = WandbLogger(project='MLOPS_SpamHam', config=hyper)
            trainer = pl.Trainer(max_epochs=epochs, logger=wandb_logger, log_every_n_steps=10)
            wandb_logger.log_hyperparams(hyper)
        else:
            trainer = pl.Trainer(
                logger=False,
                max_epochs=epochs,
            )

        #Lightning bby
        litmodel = lynModel(model, optimizer_lr=lr)
        trainer.fit(litmodel, trainloader, validloader)
        accuracy_score = trainer.validate(val_dataloaders=validloader)

        return accuracy_score[0]['val_loss']

    study = optuna.create_study(direction=cfg.sweeper.direction)
    study.optimize(optuna_objective, n_trials=cfg.sweeper.n_trials)
    params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score}\n")
    print(f"Optimized parameters: {params}\n")

if __name__ == "__main__":
    my_app()

