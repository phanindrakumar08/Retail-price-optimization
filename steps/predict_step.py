from typing import Dict, List

import numpy as np
import pandas as pd
from rich import print as rich_print
from zenml import step
from zenml.integrations.bentoml.services import BentoMLDeploymentService

@step
def predictor(
    inference_step: pd.DataFrame,
    service :  BentoMLDeploymentService,
)-> np.ndarray:
    service.start(timeout=10)
    inference_step = inference_step.to_numpy()
    prediction = service.predict("predict_ndarray", inference_step)
    rich_print("Prediction", prediction)
    return prediction