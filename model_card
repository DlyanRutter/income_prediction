Model Details

This is a Random Forest Classification model that predicts if an individual earns over 50k per year based on demographic and financial features. 
It was trained on the USL Census data, which is traditionally used for research purposes.


Intended Use
Dataset was originally created to support the machine learning community in conducting empirical analyses of ML algorithms. 
It can be used in studies of demographics and their effects on income.  

Training Data
The raw data can be found at https://github.com/DlyanRutter/income_prediction/blob/main/data/data.csv.

Evaluation Data
The preprocessed data can be found at https://github.com/DlyanRutter/income_prediction/blob/main/data/clean_data.csv During model training, 
it was split into train, test, and validation sets.

Metrics
The model was evaluated according to its F1 score, Precision, Recall, and ROC score. Its F1 score had a weighted average of 0.96. 
Its recall’s weighted average was 0.96. Its Precision was 0.96. Its ROC score was 93.01. The detailed report can be found 
in https://github.com/DlyanRutter/income_prediction/blob/main/data/figure_file/

Ethical Considerations
There are many factors that contribute to income. Demographic data alone doesn’t capture all the intricacies of life experiences.
It would be inappropriate to make concrete judgements based on these results.

Caveats and Recommendations

The dataset was severely class-imbalanced. The ratio of males to females was about 2:1. 
There are more examples of caucasians than every other race combined. The ratio of 50k or less earners to 50k+ earners was roughly 3:1. 
These imbalances were well accounted for using a label-based stratified split of the data, but it would be recommended to either find more data, 
try under sampling, or do further hyper parameter tuning. Though a grid search was already performed in the model’s production, 
it is possible better results could be achieved through further adjustment.
