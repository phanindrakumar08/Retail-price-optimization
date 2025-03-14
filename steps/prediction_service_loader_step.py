from typing import cast

from zenml import step
from zenml.integrations.bentoml.model_deployers.bentoml_model_deployer import (
    BentoMLModelDeployer
)
from zenml.integrations.bentoml.services.bentoml_deployment import (
    BentoMLDeploymentService,
)

@step(enable_cache= False)
def bentoml_prediction_service_loader(
    pipeline_name: str, step_name:str, model_name:str
) -> BentoMLDeploymentService:
    
    model_deployer = BentoMLModelDeployer.get_active_model_deployer()

    services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        model_name=model_name,
    )

    if not services:
        raise RuntimeError(
            f"No BentoML prediction server deployed by the"
            f" '{step_name}'  step in the '{pipeline_name}'"
            f"pipeline for the '{model_name}' model is currently"
            f"running"
        )
    
    if not services[0].is_running:
        raise RuntimeError(
            f"No BentoML prediction server deployed by the"
            f" '{step_name}'  step in the '{pipeline_name}'"
            f"pipeline for the '{model_name}' model is currently"
            f"running"
        )
    return cast(BentoMLDeploymentService, services[0])