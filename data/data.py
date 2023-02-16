import pandas as pd
import requests
import io
import pathlib 
import os 

url = "https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/data/census.csv?raw=True"
local_path = '/Applications/python_files/income_prediction/data/'#data.csv'
#url2 = 'https://github.com/DlyanRutter/nyc_airbnb/blob/main/data/sample1.csv?raw=True'
def download(url=url, local_path=local_path, filename='data.csv', online=False):
	"""
	Retrieve CSV file and return pandas data frame. 
	Url is path to csv file. 
	Local path is path to save file to. 
	Filename is name of file to save as
	Online true if you want to use the online file rather than a local version
	"""
	#basename = pathlib.Path(url).name.split("?")[0].split("#")[0]
	data_file = local_path + filename
	if not os.path.exists(data_file) and online==False:
		with open(data_file, 'wb+') as data:
			with requests.get(url, stream=True) as raw_data:
				for chunk in raw_data.iter_content(chunk_size=8192):
					data.write(chunk)
				data.flush()
			print ("file saved locally")
	elif online==True:
		request = requests.get(url).content
		return pd.read_csv(io.StringIO(request.decode('utf-8')))
	try:
		return pd.read_csv(data_file)
	except FileNotFoundError:
		return "Data file not found"
	#print ('file saved locally')
	#elif os.path.exists(data_file):
	#	data = data_file
	#else:
	#	request = requests.get(url).content
	#	data = io.StringIO(request.decode('utf-8'))

df = download()
print (df.head())