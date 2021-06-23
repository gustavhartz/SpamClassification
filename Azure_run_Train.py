import azureml.core
from azureml.core import Workspace

from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.widgets import RunDetails
import torch

from src.models.train_model import TrainOREvaluate

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# Create a Python environment for the experiment
sklearn_env = Environment("sklearn-env")

# Ensure the required packages are installed (we need pip, scikit-learn and Azure ML defaults)
packages = CondaDependencies.create(conda_packages=['pip', 'scikit-learn', 'pytorch', 'torchvision'],pip_packages=['azureml-defaults', 'transformers', 'wandb', 'matplotlib', 'pytorch_lightning', 'tqdm', 'absl-py'])

sklearn_env.python.conda_dependencies = packages

print('environment {} loaded'.format(sklearn_env.name))

# Create a script config
script_config = ScriptRunConfig(source_directory='',script='src/models/train_model.py', arguments=['lightning'], environment=sklearn_env) 


# submit the experiment run
experiment_name = 'spam_ham_experiment_2'
experiment = Experiment(workspace=ws, name=experiment_name)
print("Starting experiment:", experiment.name)

run = experiment.submit(config=script_config)

#model_file = 'sh_model.pkl'
#run.upload_file(name = 'outputs/' + model_file, path_or_stream = './' + model_file)

# Complete the run
#run.complete()

# Register the model
#run.register_model(model_path='outputs/diabetes_model.pkl', model_name='diabetes_model',tags={'Training context':'Inline Training'},properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']})
# Block until the experiment run has completed
run.wait_for_completion()

