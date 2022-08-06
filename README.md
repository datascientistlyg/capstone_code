# capstone_code
## Project Description:
This project includes a blog at https://github.com/datascientistlyg/capstone/blob/main/index.md to show how new factors and machine learning algo could increase the R2 score. 

This application could be used to as a basic script for further development to enhance the r2score of different AI models which are attracted to quantitative hedge funds.  

## Dataset Download:
Download dataset from https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data, and unzip to directory "optiver-realized-volatility-prediction" 

## File Description:
optiver_dataset_r2score.py : The script is used to calculate the naive r2score provided by optiver, linear regression model based r2score and lgb model based r2score

## File architecture:
- optiver-realized-volatility-prediction

|- book_train.parquet  # Contains "book train" orderbook parquet dataset

|- book_test.parquet  # Contains "book test" orderbook parquet dataset

|- trade_train.parquet  # Contains "trade train" orderbook parquet dataset

|- trade_test.parquet  # Contains "trade test" orderbook parquet dataset

|- train.csv  # target(train) label y dataset

|- test.csv  # target(test) label y dataset

- optiver_dataset_r2score.py

- README.md

## Run Instructions:

1. Download dataset from https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data, and unzip to directory "optiver-realized-volatility-prediction" 

2. Run your scripy: `python optiver_dataset_r2score.py`

## Reference:
1. Basic functions and algos such as "log_return", "realized_volatility", "realized_volatility_per_time_id", "past_realized_volatility_per_stock" and etc. are provided by optiver https://www.kaggle.com/code/jiashenliu/introduction-to-financial-concepts-and-data/notebook.
2. The head picture on blog is from https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data
