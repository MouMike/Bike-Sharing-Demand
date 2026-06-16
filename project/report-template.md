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

TODO: Replace the image below with your own.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](img/model_test_score.png)

## Summary
TODO: Add your explanation
