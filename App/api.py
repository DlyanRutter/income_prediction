import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from car_evaluation_model import __version__ as model_version
from car_evaluation_model.predict import make_prediction
from loguru import logger

from app import __version__, schemas
from app.config import settings

"""
The health endpoint is quite straightforward. 
It returns the health response schema of the model when you access the web server (Figure 4).
 You defined this schema in the health.py module in the schemas directory. 

The predict endpoint is slightly more complex. Here are the steps involved: 

Take the input and convert it into a pandas DataFrame: the jsonable_encoder returns a JSON compatible version of the pydantic model.   
Log the input data for audit purposes. 
Make a prediction using the ML model’s make_prediction function. 
Catch any errors made by the model. 
Return the results, if the model has no errors.

"""
api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleCarTransactionInputData) -> Any:
    """
    Make predictions with the Fraud detection model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    # Advanced: You can improve performance of your API by rewriting the
    # `make prediction` function to be async and using await here.
    logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = make_prediction(inputs=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results