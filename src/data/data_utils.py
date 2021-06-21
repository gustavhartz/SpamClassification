import torch
import pandas as pd
from torch.utils.data import Dataset


class SPAMorHAMDataset(Dataset):
    """SPAM or HAM data"""

    def __init__(self, csv_file, input_dim, tokenizer=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.frame = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = input_dim

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        message = [self.frame.iloc[idx, 1]]
        label = torch.Tensor([self.frame.iloc[idx, 0]])

        if self.tokenizer:
            message = self.tokenizer(
                message,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]

        sample = {"message": message.squeeze(), "label": label}
        return sample
