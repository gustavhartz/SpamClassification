# SpamClassification

This project is about classifying emails/sms texts as *spam* or *ham* using deep learning with a primary focus on
applying MLOPS principles to the task. Given that the data for this task is text data we will be utilizing
the [huggingface transformers](https://huggingface.co/transformers/) library. This libraray provides pretrained
tokenizer modules that eliminates the need to develop our own text preprocessing, and instead focus on the ML Ops
apsects of the implementation.

Data is collected from the
Kaggle [Spam Text Message Classification] (https://www.kaggle.com/team-ai/spam-text-message-classification) dataset.
This data is a collection of personal text messages and include many informal words.

We will use an LSTM network as our classifier, as this type of model can be very good at handling sequential data
because of it's recurrent structure.

Group members:

- Simon Jacobsen, s152655
- Jakob Vexø, s152830
- Morten Thomsen, s164501
- Gustav Hartz, s174315

## Major Frameworks and principles applied

### OPTUNA & HYDRA

For version controlling and ensuring reproducible results we have been applying the hydra framework to our pytorch
lightning framework.

**OPTUNA:** An open source hyperparameter optimization framework to automate hyperparameter search and we use it for baysian
grid search using evolutionary algorithms. This is configured using the config_hydra_optuna file.

**HYDRA:** Hydra is an open-source Python framework that simplifies the development of research and other complex
applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and
override it through config files and the command line. The name Hydra comes from its ability to run multiple similar
jobs - much like a Hydra with multiple heads.

**Pytorch Lightning:**

**Weights & Biases:** Is used for visualizations of training and is implemented as the logger in pytorch lightning. It's
primary purpose is a tracking the progress of model training. WandB can do hyperparameter sweeps, but we decided to
focus on HYDRA and OPTUNA

**Data Drifting:**

**CI/CD:** Pytest are run for the entire pytest directory "./tests". Furthermore, we also have actions for monitoring that the
commits live up to the PEP8 standard. This is done with Flake8 and isort. 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    │── deployment         <- Scripts for deploying the model as an Azure endpoint 
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    │
    ├── tests              <- pytests using the suggested src layout from pytest documentation
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------