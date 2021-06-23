from azureml.core import Environment
from azureml.core import Model
from azureml.core import Workspace
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

env = Environment(name="project_environment")
python_packages = ['torch', 'transformers', 'azureml-core']
for package in python_packages:
    env.python.conda_dependencies.add_pip_package(package)

dummy_inference_config = InferenceConfig(
    environment=env,
    source_directory="./source_dir",
    entry_script="./prediction_server.py",
)

deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1, memory_gb=1
)

ws = Workspace.from_config()
model = ws.models['spamham']
print(model.name, 'version', model.version)

DEPLOY = True
if DEPLOY:
    service = Model.deploy(
        ws,
        "spamham",
        [model],
        dummy_inference_config,
        deployment_config,
        overwrite=True,
    )
    service.wait_for_deployment(show_output=True)
