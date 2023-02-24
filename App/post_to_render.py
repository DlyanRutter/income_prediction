from fastapi.testclient import TestClient
from fastapi import status, FastAPI
import requests
import json
from App.data import process_data, download, split
from App.app import app ######Must import the app that was running in original file rather than create a new one, made issue with get

def get_salary():
	data= {
	"age": 26,
	"workclass": "private",
	"fnlgt": 189765,
	"education": "Bachelors",
	"education-num": 16,
	"marital-status": "Married-civ-spouse",
	"occupation": "Exec-managerial",
	"relationship": "Husband",
	"race": "White",
	"sex": "Male",
	"capital-gain": 592,
	"capital-loss": 88,
	"hours-per-week": 60,
	"native-country": "United-States"}
	request = requests.post('fastapi-app.onrender.com/predict_salary', auth=('usr', 'pass'), data=json.dumps(data))
  assert request.status_code == status.HTTP_200_OK
	assert request.json() == {"salary": ">50k"}
  print (request.json())
