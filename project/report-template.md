# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### MICHAEL MATELA

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
Before submitting the predictions, all negative predictions had to be set to zero because Kaggle does not accept negative values for "count", which realistically makes sense.

### What was the top ranked model that performed?
The top ranked model that performed was 'WeightedEnsemble_L2'.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
The exploratory analysis found the distribution of data for each feature. This was achieved by generating a histogram for each feature. Exploratory data analysis was also performed on the dataset through the 'train.describe()' command. Below are some of the findings:
- datetime - demand was generally equally distributed across the different dates.
- season - demand was very consistent across all seasons, with slightly lower demand in Spring than in other seasons. 
- holiday - the demand was almost entirely on non-holiday days. This is in most likelihood because there are only a few holidays in a year.
- workingday - approximately 60% of the demand was on working days. In a five-day working week, this means that the demand, per day, on weekends is significantly higher on weekends and holidays than on working days.
- weather - highest demand was experienced on clear/partly cloudy days, followed by misty days with some clouds, and the lowest demand recorded on days with heavy precipitation.
- temp - demand was approximately evenly distributed during temperatures ranging from 10 degrees celsius to 30 degrees celsius with significantly lower demand outside this range.
- humidity - reasonable demand was experienced across all levels of humidity above 30% with the highest demand recorded for humidity ranging from 40% to 90%.
- windspeed - demand showed a significantly high preference for low wind speeds.
- casual - only a small number of non-registered user rentals were initiated.
- registered - no registered user rentals were initiated on most days, however, there was significantly more registered user rentals were initiated compared to non-registered user rentals.

### How much better did your model preform after adding additional features and why do you think that is?
The model perfomed significantly better after adding an additional feature, increasing the best model's score_val from '-121.8' to '-30.9'. The Kaggle score also showed significant improvement, moving from '1.41128' to '0.48827'.
The additional feature reduced the bias in the model, and therefore, improving the predictions.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
The model performance worsened. Increasing the number of iterations to a very high number improved the predictions/scores but minimally. The predictions were still worse off compared to those prior to hyperparameter tuning. My two key takeaways were the following:

- Autogluon hyperparameters are already very well optimised and, therefore, it is very difficult to beat the performance of the models by tuning the hyperparameters manually.
- Tuning the hyperparameters manually limits them to specific values and also, autogluon is limited to only the models that are listed. Autogluon utilises many models and many hyperparameters and only selecting a few models, and tuning a few hyperparameters, constraints autogluon from performing to its fullest capability.

### If you were given more time with this dataset, where do you think you would spend more time?
I would spend more time feature engineering. Adding a new feature to the dataset drastically improved the predictions and, therefore, it would likely be benficial to discover which other new features to add.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|100|31|0.1|1.41128|
|add_features|100|31|0.1|0.48827|
|hpo|3000|100|0.05|0.52200|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](img/model_test_score.png)

### Model performance comparison.
The best performing algorithm was the Tabular Deep Learning Model (TABM) which utilizes neural networks to produce predictions. Second was the the Categorical Boosting algorithm (CAT). CAT is known to be good with categorical datasets. The dataset contained to categorical features, namely, weather and season, and categorising these two features helps an algorithm like CAT to perform better on the data. The third best performing algorithm wat the Gradient Boosting Machine algorithm (GBM) which is a tree ensemble where trees are built sequentially and each new tree improving the performance of the previous trees. GBM has an edge over Random Forest (RF)because RF builts trees in parallel meaning that the trees do not improve each others performances.

The worst performing algorithm was the Linear Regression algorithm (LR) and the K-Nearest Neighbors (KNN) was the second worst. The very poor perfomance by the LR is an indication that the data points are non-linear. KNN is a classification algorithm and therefore it is expected to perform poorly on a regression problem.

The figure below shows the visual comparison of the models, where the high negative scores indicate the worst performing algorithms and the low negative scores (closer to zero) indicate the best perfoming models.

![model_score_comparison.png](img/model_score_comparison.png)

## Summary
The purpose of this project was to use the AutoGluon library to train several machine learning models for the Bike Sharing Demand on Kaggle. AutoGluon is an AutoML framework and its strength is in its ability to train and ensemble multiple models in order to produce high-accuracy predictions.

The first step was to input raw data into the AutoGluon TablePredictor with default parameters and to output a set of initial predictions. The best-ranked model was the 'WeightedEnsemble_L2' and, therefore, this was the model that AutoGluon used to produce predictions. Before submitting the predictions on Kaggle for scoring, it was necessary to set the negative values to zero because it did not make sense to have a negative 'count'. The Kaggle score was the retrieved to check how the model performed.

Several modifications, such as feature engineering, hyperparameter tuning and change in the evaluation, were performed in an attempt to improve the quality of the predictions, and the following findings were made:
- adding a new 'month' feature greatly improved the top model score (from '-121.8' to '-30.9') and the Kaggle score (from '1.41128' to '0.48827').
- manually modifying the hyperparameters worsened the model performances. This highlights how well optimised AutoGluon hyperparameters are, and also, the high level of skill that is required to improve, or beat, its performance.
- changing the evaluation metric only slightly altered the model performances. 
