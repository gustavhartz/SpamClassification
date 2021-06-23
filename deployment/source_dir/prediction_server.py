import json
import logging
from transformers import AutoTokenizer
import torch
import os
import logging
from model import LSTM

log = logging.getLogger(__name__)
tokenizer = ""
model = ""
tokenizer_config = {}


def init():
    global model
    global tokenizer
    global tokenizer_config
    log.info("Started")
    model = LSTM(28996)
    model.load_state_dict(torch.load("source_dir/model_state"))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenizer_config = {"padding": "max_length", "truncation": True, "max_length": model.text_len,
                        "add_special_tokens": False,
                        "return_tensors": "pt"}


def run(data):
    data_processed = tokenizer(json.loads(data), **tokenizer_config)["input_ids"]
    output = model(data_processed)
    preds = torch.zeros(output.shape)
    preds[output >= 0.15] = 1
    res = list(map(lambda x: {0: "HAM", 1: "SPAM"}.get(x), preds.view(-1).numpy().astype(int).tolist()))
    return json.dumps(res)
