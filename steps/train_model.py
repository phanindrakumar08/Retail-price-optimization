import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from statsmodels.regression.linear_model import RegressionResultsWrapper
from typing import List
from typing import Annotated, Tuple
from zenml import step
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from materializers.custom_materializer import(
    ListMaterializer, 
    SKLearnModelMaterializer,
    StatsModelMaterializer,
)
from steps.src.model_building import LinearRegressionModel, ModelRefinement
 
experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "your active stack needs to contain a MLFlow experiment tracker for this example to work"
    )
@step(experiment_tracker = "mlflow_tracker",
      settings = {'experiment_tracker.mlflow':{"experiment_name":"test_name"}},
      enable_cache = False, output_materializers = [SKLearnModelMaterializer, ListMaterializer])

def sklearn_train(
    X_train: Annotated[pd.DataFrame, "X_train"],
    y_train: Annotated[pd.DataFrame, "y_train"]
    )-> Tuple[
        Annotated[LinearRegression, "model"],
        Annotated[List[str], "predictors"],
    ]:
    try:
        mlflow.end_run()
        with mlflow.start_run() as run:
            mlflow.sklearn.autolog()
            model = LinearRegression()
            model.fit(X_train, y_train)

            predictors = X_train.columns.tolist()
            return model, predictors
    except Exception as e:
        raise e
    

#pip install bentoml>=1.0.10

@step(experiment_tracker="mlflow_tracker",
  settings={"experiment_tracker.mlflow": {"experiment_name": "test_name"}}, output_materializers=[StatsModelMaterializer, ListMaterializer])
def re_train(
    X_train: Annotated[pd.DataFrame, "X_train"],
    y_train: Annotated[pd.Series, "y_train"], 
    predictors: list
) -> Tuple[
    Annotated[RegressionResultsWrapper, "model"],
    Annotated[pd.DataFrame, "df_with_significant_vars"],
]:
    """Trains a linear regression model and outputs the summary."""
    try:  
        print(X_train[predictors])
        model = LinearRegressionModel(X_train[predictors], y_train)
        mlflow.statsmodels.autolog()
        model = model.train()  # Train the model
        df_with_significant_vars = pd.concat([X_train[predictors], y_train], axis=1)  
        df_with_significant_vars.rename(columns={"series": 'qty'}, inplace=True) 
        # df_with_significant_vars.to_csv("df_with_significant_vars.csv", index=False)
        logger.info("Model trained successfully")
        return model, df_with_significant_vars  # Return the model and predictors
    except Exception as e:
        logger.error(e)
        raise e
