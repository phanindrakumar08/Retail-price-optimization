from zenml import pipeline
from zenml.logger import get_logger
from zenml.integrations.constants import BENTOML
from steps.deployment_trigger_step import deployment_trigger
from steps.evaluator import evaluate
from steps.bento_builder import bento_builder
from steps.deployer import bentoml_model_deployer

logger = get_logger(__name__)
from zenml.client import Client

print(Client().active_stack.experiment_tracker.get_tracking_uri())


from zenml.config import DockerSettings


from steps.data_splitter import combine_data, split_data
from steps.ingest_data import ingest
from steps.process_data import categorical_encode, feature_engineer
from steps.train_model import re_train, sklearn_train

docker_settings = DockerSettings(required_integrations=[BENTOML])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def training_retail():
    """Train a model and deploy it with BentoML."""
    df = ingest("retail_prices") 
    df_processed = categorical_encode(df)
    df_transformed = feature_engineer(df_processed)  
    X_train, X_test, y_train, y_test = split_data(df_transformed)  
    model, predictors = sklearn_train(X_train, y_train)         # Evaluate model
    rmse = evaluate(model, df)
    decision = deployment_trigger(accuracy = rmse, min_accuracy = 0.80)
    bento = bento_builder(model = model)
    bentoml_model_deployer(bento= bento, deploy_decision = decision)
