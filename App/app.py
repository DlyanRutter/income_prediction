from fastapi import FastAPI, HTTPException
import joblib, numpy
from pydantic import BaseModel, Field 
from fastapi.encoders import jsonable_encoder
import pandas as pd
from data import process_data

class Item(BaseModel):
    """
    Define inputs and types
    Modify column names with hyphens for compatibility
    """
    age: int
    workclass: object
    fnlgt: int
    education: object
    education_num: int = Field(alias="education-num") 
    marital_status: object = Field(alias="marital-status")
    occupation: object
    relationship: object
    race: object
    sex: object
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: object = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True

# Load models, they were stored in the prior directory
cv_rfc = joblib.load('rfc_model.pkl')
lrc = joblib.load('logistic_model.pkl')

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route, return json of message
@app.get("/")
async def root():
    return {"Message": "This is a Salary Prediction Model API"}

# Define the route to the sample predictor
@app.post("/predict_salary") 
async def predict_salary(sample: Item): 
    """
    Input a instance of raw data
    Return value is a salary prediction
    """
    # Confirm input sample is valid
    if(not(sample)): 
        raise HTTPException(status_code=400, 
                            detail="Please Provide a valid sample")
    # jsonable_encoder converts BaseModel object to json
    answer_dict = jsonable_encoder(sample)
    salary = "" 

    for key, value in answer_dict.items():
        answer_dict[key] = [value]
    person = pd.DataFrame.from_dict(answer_dict) # Make df   
    person = process_data(person) # Process data for model compatability
    prediction = cv_rfc.predict(person) # Predict on created df

    # Determine person's salary prediction
    if(prediction[0] == 0):
        salary = ">50k" 

    elif(prediction[0] == 1):
        salary = "<=50k" 
        
    return {
            "salary": salary 
           }
