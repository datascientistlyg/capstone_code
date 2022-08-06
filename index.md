
## Exploring new factors and machine learning algorithm to increase the R2 score

### Problem Introduction
I am really interested in trading industry and always interested what causing the change of the rate of return and R2 score the key indicator of strategy performance.

![image](https://user-images.githubusercontent.com/109795677/183232847-91323839-b36f-4d0d-a5d6-76a95892cbab.png)
*) Picture took from (https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/overview)

### Strategy to solve the problem
I did analysis using the dataset downloaded from [Kaggle](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data). And I solve the prolem through found the answers to the following questions:

1. What is the R2 score of naive prediction provided by Optiver?
2. Which new factors could enhance the in sample R2 score by linear regression?
3. Can lightgbm further increase the in sample R2 score?

### Metrics

R2 Score is widely used as the key metrics of multifactor strategy, hence the goal of this script is to find ways to increase the r2score

### EDA
"bid ask spread","bid gap","ask gap", "bid size imbalance", "ask size imbalance" are added to the features. Following is the feature importance graph for reference.

### Modeling

Model 1. Naive prediction used by optiver as example
Model 2. Linear Regression with new factors
Model 3. Lightgbm based models

### Hyperparameter Tuning

Since the goal is to increase the r2 score, we can use large max_bin, smaller learning_rate, larger num_leaves to increase the accuracy. Here learning rate = 0.01 is used to tune the model.

### Results

#### R2 score of naive prediction provided by Optiver

![image](https://user-images.githubusercontent.com/109795677/183234385-22a792b7-183e-44d6-abc9-92ce4dbcdbf6.png)

The picture above show the naive prediction scatter graph provided by Optiver. The R2 score is 0.628

#### New factors which enhance the in sample R2 score by linear regression
![image](https://user-images.githubusercontent.com/109795677/183234460-e088c74e-81e9-4943-9a67-6b69ce17f4a9.png)

The graph shows that the orange dots based on linear regression with new factors("bid ask spread","bid gap","ask gap", "bid size imbalance", "ask size imbalance") are more converge than the blue ones which is provided by naive prediction. The R2 score of linear regression is 0.775

#### lightgbm further increases the in sample R2 score?
![image](https://user-images.githubusercontent.com/109795677/183235170-82164117-1096-4a91-9ca1-1330c051e737.png)

From the picture above, we can see that lightgbm model in green dots is more converge than the linear model. The R2 score of lightgbm regression is 0.817 

### Conclusion
New features such as "bid ask spread","bid gap","ask gap", "bid size imbalance", "ask size imbalance" with linear regression model can increase the R2 score and models like lightgbm could further increase the R2 score.

### Improvements
[Source Code](https://github.com/datascientistlyg/capstone_code).
