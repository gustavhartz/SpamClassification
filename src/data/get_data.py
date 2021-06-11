import click
import logging
from pathlib import Path
from torchvision import transforms
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np


@click.command()
@click.option('--process_data', default=True, help='Apply  to the raw dataset')
def main(process_data):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Collecting the dataset')
    # Assume data
    if Path('./data/processed/train_set.csv').exists():
        print("Data already downloaded.")
    else:
        #find and load environment file
        load_dotenv(find_dotenv())
        import kaggle
        #authenticates kaggle user using environment file
        kaggle.api.authenticate()
        #downloads and unzips to raw folder
        
        kaggle.api.dataset_download_files('team-ai/spam-text-message-classification', path = 'data/raw', unzip=True)
        #time.sleep(1)
        print("Succesfully downloaded kaggle data")

    #split data into train/test/val
    raw_data = pd.read_csv('data/raw/SPAM text message 20170820 - Data.csv')
    np.random.seed(404)
    msk = np.random.rand(len(raw_data)) < 0.8
    msk_ = np.random.rand(len(raw_data[msk])) < 0.8

    train_data = raw_data[msk][msk_]
    val_data = raw_data[msk][~msk_]
    test_data = raw_data[~msk]

    train_data.to_csv('data/processed/train_set.csv')
    val_data.to_csv('data/processed/val_set.csv')
    test_data.to_csv('data/processed/test_set.csv')
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()