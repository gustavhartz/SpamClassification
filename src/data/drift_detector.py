import matplotlib.pyplot as plt
import torch
import torchdrift
from sklearn.manifold import Isomap
from transformers import AutoTokenizer

from src.data.data_utils import SPAMorHAMDriftset


def get_email_spam(file, tokenizer, tokenizer_dim):
    texts = []
    text = ""
    start = False
    ct = 0
    start_next = False
    with open(file, "r", encoding="latin-1") as file:

        for row in file:

            if start_next:
                start = True
                start_next = False

            if ct > 500:
                break

            if row.startswith("From r"):
                texts.append(text)
                text = ""
                ct += 1
                start = False
            elif row.startswith("Status: "):
                start_next = True

            if start:
                if row.strip() != "":
                    text += " " + row.replace("\n", "")

    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer_dim,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]


def run_drifter(model_path):
    class feature_extractor(torch.nn.Module):
        def __init__(self, model):
            super(feature_extractor, self).__init__()
            self.model = model

        def forward(self, batch):
            x = self.model.embedding(batch)
            output, _ = self.model.lstm(x)

            return output.reshape(output.shape[0], -1).detach()

    model = torch.load(model_path)
    feature_extractor = feature_extractor(model)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    train_set = SPAMorHAMDriftset(
        "./data/processed/train_set.csv",
        input_dim=model.text_len,
        tokenizer=tokenizer,
    )

    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
    )

    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
    torchdrift.utils.fit(trainloader, feature_extractor, drift_detector)

    # Run on new data
    texts = get_email_spam("./data/raw/fradulent_emails.txt", tokenizer, model.text_len)
    features = feature_extractor(texts)
    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)

    # N_base = drift_detector.base_outputs.size(0)
    mapper = Isomap(n_components=2)
    base_embedded = mapper.fit_transform(drift_detector.base_outputs)
    features_embedded = mapper.transform(features)
    plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c="r")
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    plt.title(f"score {score:.2f} p-value {p_val:.2f}")
    plt.show()


if __name__ == "__main__":
    model_path = "outputs/2021-06-23/13-00-54/model.model"

    run_drifter(model_path)
