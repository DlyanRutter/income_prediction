import io
import pathlib
import joblib
import pandas as pd
import numpy as np
import requests
import io
import pathlib
import joblib
import os
import logging
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt
import seaborn as sns


logging.basicConfig(
    filename='./data_logs.log',  # Path to log file
    level=logging.INFO,  # Log info, warnings, errors, and critical errors
    filemode='a',  # Create log file if one doesn't already exist and add to existing log with each run
    format='%(asctime)s-%(name)s - %(levelname)s - %(message)s',
      # Format log string
    datefmt='%d %b %Y %H:%M:%S %Z',  # Format date
    force=True)  # Create separate log files, basic_config only configures unconfigured root files
logger = logging.getLogger()

url = "https://github.com/DlyanRutter/income_prediction/blob/main/data/data.csv?raw=True"
local_path = '/Applications/python_files/income_prediction/data/'


def download(
    url=url,
    local_path=local_path,
    filename='data.csv',
     online=False):
	"""
	Retrieve CSV file and return pandas data frame.
	Url is path to csv file.
	Local path is path to data.py
	Filename is name of file to save as
	Online true if you want to use the online file rather than a local version
	"""
	# basename = pathlib.Path(url).name.split("?")[0].split("#")[0]
	data_file = local_path + filename
	# Save file to local machine if file doesn't yet exist
	if not os.path.exists(data_file) and online == False:
		with open(data_file, 'wb+') as data:
			with requests.get(url, stream=True) as raw_data:
				for chunk in raw_data.iter_content(chunk_size=8192):
					data.write(chunk)
				data.flush()
			logger.info(f"File created saved locally as {data_file}...")

	# Pull file from github but don't save
	elif online == True:
		request = requests.get(url).content
		logger.info(f"Using github version of data")
		return pd.read_csv(
    io.StringIO(
        request.decode('utf-8')),
         skipinitialspace=True)

	# Load csv file and convert to data frame
	try:
		logger.info('using local file')
		return pd.read_csv(data_file, skipinitialspace=True)
	except FileNotFoundError:
		logger.info("Data file not found")


#data_frame = download()


def process_data(df):
	# remove whitespace from column names for easier indexing
	df.columns = df.columns.str.replace(' ', '')
	# drop duplicates
	logger.info("Dropping duplicates")
	df = df.drop_duplicates().reset_index(drop=True)

	# Binarize labels
	df['salary'] = df['salary'].apply(
        lambda val: 0 if val == ">50K" else 1)

	# Dropping capital gain outliers [There were 159 at exactly 99999]
	idx = df['capital-gain'].between(0, 99998)
	df = df[idx].copy()

	# Specify numeric and categorical features
	cat_features = ["workclass", "education", "marital-status", "occupation",
	"relationship", "race", "sex", "native-country"]
	num_features = ['age', 'fnlgt', 'education-num', 'capital-gain',
    'capital-loss', 'hours-per-week']

    # Encode categorical features
	for cat in cat_features:
		df[cat] = df[cat].astype('category')
		df[cat] = df[cat].cat.codes

    # Impute missing values for numeric and categorical features 
	imputer = SimpleImputer(strategy="most_frequent")
	df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

	salary = df['salary']
	df = df.drop('salary', axis=1)  
	# Standardize numerical data
	df_z_scaled = df.copy()
	for num in num_features:
		df_z_scaled[num] = \
		(df_z_scaled[num] - df_z_scaled[num].mean()) / df_z_scaled[num].std()
	#df.to_csv('clean_data', index=False)
	df_z_scaled['salary'] = salary
	return df_z_scaled

#data_frame = process_data(data_frame)


def slices(feature, df, classes=None, property=None):
	"""
	Feature: a categorical feature from the list:
		["workclass", "education", "marital-status", "occupation",
		"relationship", "race", "sex", "native-country"]	
	Property: The feature's numerical property to analyze from the list:
		["eduction-num", "capital-gain", "capital-loss", "age",
		"fnlgt", "hours-per-week"]
	Classes: a list of potential values for the categorical feature, e.g.
		["Federal-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc",
		"Self-emp-not-inc", "?", "State-gov", "Without-pay"] for workclass feature
	
	"""
	# Show mean possible numeric values of all potential workclass categoricals
	"""
		                        age          fnlgt  ...  capital-loss  hours-per-week
	workclass                                   ...                              
	?                 40.960240  188516.338235  ...     60.760349       31.919390
	Federal-gov       42.590625  185221.243750  ...    112.268750       41.379167
	Local-gov         41.751075  188639.712852  ...    109.854276       40.982800
	"""
	mean_df = df.groupby(feature).mean(numeric_only=True) 
	std_df = df.groupby(feature).std(numeric_only=True) 

	if not property and not classes:
		print (mean_df)
		print (std_df)

	if property and not classes:
		print (mean_df[property])
		print (std_df[property])

	if property and classes:
		for cls in classes:
			std = std_df.loc[cls][property]
			mn = mean_df.loc[cls][property]
			print (f"standard deviation of {property} of {cls} = {std}")
			print (f"mean {property} of {cls} = {mn}")

	if classes and not property:
		raise ValueError("Can't have a property without a class")
	#slices('workclass', data_frame, classes=['Never-worked', 'Federal-gov'], property='education-num')


def split(df, stratify_by=None):

	#df is a data frame to split into train and test sets
	#stratify_by is str of the feature to stratify by if provided, else none
	#e.g. "Workclass" 
	copied_df = df.copy()
	y = copied_df.pop('salary') #labels
	X = copied_df #features
	X_train, X_val, y_train, y_val = train_test_split(X, y,
		test_size=0.25, stratify=X[stratify_by] if stratify_by else None, random_state=42)
	return X, y, X_train, X_val, y_train, y_val

def train_models(df, stratify_by=None):
	'''
	Trains logistic regression and random forest models
	input:
		features_train: features training data
		features_test: features testing data
		labels_train: labels training data
		labels_test: labels testing data
	output:
		None
		Cell may take up to 15-20 minutes to run
	'''
	# Create file to store model
	filename = '/models'
	current_dir = os.getcwd()
	models = current_dir + filename
	if not os.path.isdir(models):
		os.umask(0)
		os.makedirs(models)
		if not os.path.exists(current_dir + filename):
			with open(models, 'w', encoding='utf-8'):
				pass

	y = df.pop('salary') #labels
	X = df 
	features_train, features_test, labels_train, labels_test = train_test_split(X, y,
		test_size=0.3, stratify=y, shuffle=True, random_state=42)# if stratify_by else None, random_state=42)
	# Create Random Forest classifier and logistic regression classifier
	rfc=RandomForestClassifier()
	lrc=LogisticRegression(solver ='lbfgs', max_iter=3000)

	# Grid search for random forest classifier
	param_grid={
	    'n_estimators': [200, 500],
	    'max_depth' : [4,5,100],
	    'criterion' :['gini', 'entropy'],
	    'max_features': ['sqrt']}
	cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

	# Fit both classifiers
	cv_rfc.fit(features_train, labels_train)
	lrc.fit(features_train, labels_train)

	# Save best models
	joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
	joblib.dump(lrc, './models/logistic_model.pkl')
	print ('success, finally')

def classification_report_image(labels_train, labels_test, y_train_preds_lr,
    y_train_preds_rf, y_test_preds_lr, y_test_preds_rf):
    '''
    Produces classification report for logistic regression and random forest
    classifiers and stores as a compounded image in images folder
    input:
        labels_train: training response values
        labels_test: test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest
    output:
        None
    '''
    # Create file 'figure_file' in current directory to save and store figures
    filename = '/figure_file'
    current_dir = os.getcwd()
    figure_file = current_dir + filename
    if not os.path.isdir(figure_file):
        #path = os.path.join(current_dir, filename)
        os.umask(0)
        os.makedirs(figure_file)
    if not os.path.exists(current_dir + filename):
        with open(figure_file, 'w', encoding='utf-8'):
            pass
    # Plot random forest classification report
    plt.rc('figure', figsize = (13, 6))
    plt.text(0.4, 1.25, str('Random Forest Train'), {
        'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.4, 0.05, str(classification_report(labels_test, y_test_preds_rf)), {
        'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.4, 0.6, str('Random Forest Test'), {
        'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.4, 0.7, str(classification_report(labels_train, y_train_preds_rf)), {
        'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')

    # Plot Logistic Regression report
    plt.rc('figure', figsize = (13, 6))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {
        'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(labels_test, y_test_preds_lr)), {
        'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
        'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(labels_train, y_train_preds_lr)), {
        'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./figure_file/classification_reports.png')
    plt.close()

def roc_plots(labels_test, y_test_preds_rf, y_test_preds_lr):
    '''
    Create superimposed ROC plots for random forest and logistic regression
    and stores in impages file
    Inputs:
        labels_test: test label values
        y_test_preds_rf: random forest test predictions
        y_test_preds_lr: logistic regression test predictions
    Output:
        None
    '''
    filename = '/figure_file'
    current_dir = os.getcwd()
    figure_file = current_dir + filename
    if not os.path.isdir(figure_file):
        #path = os.path.join(current_dir, filename)
        os.umask(0)
        os.makedirs(figure_file)
    if not os.path.exists(current_dir + filename):
        with open(figure_file, 'w', encoding='utf-8'):
            pass
    # Plot logistic regression roc curve
    lr_roc = RocCurveDisplay.from_predictions(
        labels_test,
        y_test_preds_lr,
        drop_intermediate=False,
        color= "darkorange",
        name = "logistic regression classifier")

    # Plot random forest roc curve and superimpose
    rf_roc = RocCurveDisplay.from_predictions(
        labels_test,
        y_test_preds_rf,
        ax=plt.gca(),
        name = 'random forest classifier',
        color='blue')
    plt.savefig('./figure_file/roc_curves.png')
    plt.close()

def feature_importance_plot(model, x_data, output_pth='figure_file', show_plot=False):
    '''
    Create and store feature importance plot in given path
    Inputs:
        Model: Model object, random forest best_estimator_ from grid search
        Data: Cleaned Pandas dataframe of features
        output_pth: directory to store figure
        show_plot: if True, show plot upon function execution
    Output:
        None
    '''
    # Create or open file to store figure
    filename = '/figure_file'
    current_dir = os.getcwd()
    figure_file = current_dir + filename
    if not os.path.isdir(figure_file):
        os.umask(0)
        os.makedirs(figure_file)
    if not os.path.exists(current_dir + filename):
        with open(figure_file, 'w', encoding='utf-8'):
            pass
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names=[x_data.columns[i] for i in indices]

    # Create plot and title
    plt.figure(dpi = 100, figsize = (14, 7))
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars, features names as x-axis labels
    plt.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation = 90)
    # Ensure full labels show
    plt.tight_layout()
    plt.savefig(output_pth + '/feature_importances.png')
    if show_plot is True:
        plt.show()
    plt.close()

def tree_explainer_plot(model, features_test):
    '''
    Explain the output of ensemble model and store plot in image file
    Inputs:
        model: Tree model
        features_test: Features test data
    Output:
        None
    '''
    # Create or open storage file
    filename = '/figure_file'
    current_dir = os.getcwd()
    figure_file = current_dir + filename
    if not os.path.isdir(figure_file):
        os.umask(0)
        os.makedirs(figure_file)
    if not os.path.exists(current_dir + filename):
        with open(figure_file, 'w', encoding='utf-8'):
            pass

    # Create and save Shap summary plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_test)
    shap.summary_plot(shap_values, features_test, plot_type="bar", show=False)
    plt.savefig('./figure_file/tree_explainer.png')
    plt.close()

if __name__ == '__main__':
    # Load data, perform eda, encode categorical features, and engineer features
    data_frame = download()
    data_frame = process_data(data_frame)
    features, labels, features_train, features_test,\
    labels_train, labels_test = split(data_frame)

    # Train models if models not yet already created
    #if os.path.exists('./models/rfc_model.pkl'):#isdir('./models'):
    #    print ('trained models exist')
    #else:
    #    train_models(data_frame)#features_train, features_test, labels_train, labels_test)
    #    print ('models trained for the first time')
    cv_rfc = joblib.load('./models/rfc_model.pkl')
    lrc = joblib.load('./models/logistic_model.pkl')

    # Establish predictions and produce roc auc score
    labels_train_preds_rf = cv_rfc.predict(features_train)
    labels_test_preds_rf = cv_rfc.predict(features_test)
    labels_train_preds_lr = lrc.predict(features_train)
    labels_test_preds_lr = lrc.predict(features_test)
    print (roc_auc_score(labels_test, labels_test_preds_rf))

    # Create images
    classification_report_image(labels_train, labels_test, labels_train_preds_lr,
        labels_train_preds_rf, labels_test_preds_lr, labels_test_preds_rf)
    roc_plots(labels_test, labels_test_preds_rf, labels_test_preds_lr)
    feature_importance_plot(cv_rfc, features)
    #tree_explainer_plot(cv_rfc, features_test)
