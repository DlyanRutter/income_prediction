from fastapi.testclient import TestClient
from fastapi import status
import requests
import json
from income_prediction.App.data import process_data, download, split

app = FastAPI()
def test_download():
	df = download()
	assert df.shape[1] == 15

def test_process_data():
	df = download()
	df = process_data(df, save=False)
	for col_type in df.dtypes:
		valid_types = ['float64', 'int64', 'object']
		assert col_type in valid_types

def test_split():
	df = download()
	df = process_data(df)
	X, y, X_train, X_val, y_train, y_val = split(df)
	assert X_train.shape[1] == 14
	assert X_val.shape[1] == 14
	assert len(list(y_train)) == y_train.shape[0]
	assert len(list(y_val)) == y_val.shape[0]
	assert X.shape[1] == 14
	assert len(list(y)) == y.shape[0]

client = TestClient(app)

def test_get_path():
    r = client.get("/")
    assert r.status_code == status.HTTP_200_OK
    assert r.json() == {"Message": "This is a Salary Prediction Model API"}

def test_over_50k():
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
	r = client.post("/predict_salary", data=json.dumps(data))
	request = requests.post('http://127.0.0.1:8000/predict_salary', auth=('usr', 'pass'), data=json.dumps(data))
	assert request.status_code == status.HTTP_200_OK
	assert request.json() == {"salary": ">50k"}
	print(request.status_code)
	print(request.json())

def test_under_50k():
	data={
	"age": 26,
	"workclass": "private",
	"fnlgt": 18976,
	"education": "Preschool",
	"education-num": 0,
	"marital-status": "Married-civ-spouse",
	"occupation": "Exec-managerial",
	"relationship": "Husband",
	"race": "White",
	"sex": "Male",
	"capital-gain": 0,
	"capital-loss": 0,
	"hours-per-week": 0,
	"native-country": "United-States"}
	r = client.post("/predict_salary", data=json.dumps(data))
	request = requests.post('http://127.0.0.1:8000/predict_salary', auth=('usr', 'pass'), data=json.dumps(data))
	assert request.status_code == status.HTTP_200_OK
	assert request.json() == {"salary": "<=50k"}
	print(request.status_code)
	print(request.json())
test_download()
test_process_data()
test_split()
test_get_path()
test_over_50k()
test_under_50k()
