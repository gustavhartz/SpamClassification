import logging

import numpy as np
import pytest
from transformers import AutoTokenizer

from src.models.model import LSTM


def test_check_data_size():
    assert True


# Testing if model output size matches the expected size
def model_output(vocab_size, text_len, batch):

    model = LSTM(vocab_size, hidden_dimension=32, embedding_dim=10, text_len=text_len)

    return model(batch["input_ids"])


def test_model_dim():

    input_dim = 40
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    test_texts = [
        "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... "
        "Cine there got amore wat...",
        "Ok lar... Joking wif u oni...",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question"
        "(std txt rate)T&C's apply 08452810075over18's",
        "U dun say so early hor... U c already then say...",
        "Nah I don't think he goes to usf, he lives around here though",
        "FreeMsg Hey there darling it's been 3 week's now and no word back! "
        "I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv",
    ]

    batch = tokenizer(
        test_texts,
        padding="max_length",
        truncation=True,
        max_length=input_dim,
        add_special_tokens=False,
        return_tensors="pt",
    )

    assert model_output(tokenizer.vocab_size, input_dim, batch).shape == (
        len(test_texts),
        1,
    )
