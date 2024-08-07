

Time Series

Purpose
This project aims to predict stock prices using historical data and machine learning techniques. The dataset includes historical stock prices for Apple Inc. (AAPL) and Texas Instruments (TXN). The project involves data loading, preprocessing, feature selection, train-test splitting, model training, evaluation, and visualization.

----------------

How to Run This Code:

 - Ensure you have Python installed on your system along with the required packages.
 - Place AAPL.csv in the directory where the script is located.
 - Run the main.py script in a Python environment.

Dependencies:

The following libraries are required:

pandas: For data manipulation and analysis
numpy: For numerical operations
matplotlib: For plotting graphs
seaborn: For data visualization
scikit-learn: For machine learning algorithms and evaluation metrics
xgboost: For training the XGBoost model
statsmodels: For time series analysis
logging: For logging errors and information

----------------

Ensure they are installed using pip:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels

-----------------

Project Structure

├── data/
│   ├── __init__.py
│   ├── AAPL.csv
│   └── data_loader.py
|   └── data_processing.py
├── feature/
│   ├── __init__.py
|   ├──train_test_split.py
│   └── feature_selection.py
│
├── models/
│   ├── __init__.py
│   ├── arima_model.py
│   ├── arimax_model.py
│   ├── sarima_model.py
│   ├── sarimax_model.py
│   └── xgboost_model.py
│
├── prediction/
│   ├── __init__.py
│   └── backtest.py
    └──prediction.py
│
├── visualization/
│   ├── __init__.py
│   └── plots.py
├── main.py
├── requirements.txt
└── README.md

------------------

Detailed Steps
1. Data Loading
The dataset is loaded from a CSV file named AAPL.csv using the load_data function from data_loader.py. This function reads the data into a pandas DataFrame and performs initial preprocessing.
2. Data Preprocessing
Preprocess Data: The preprocess_data function in data_processing.py handles the data preprocessing steps including creating target variables.
3. Feature Selection
Select Features: The select_features function in feature_selection.py selects the relevant features for model training.
4. Data Splitting
Train-Test Split: The create_train_test_split function in train_test_split.py splits the data into training and testing sets.
5. Data Visualization
Plot Line: Visualize the stock price over time using the plot_line function.
Plot Seasonal Decomposition: Visualize the seasonal decomposition of the stock price using the plot_seasonal_decompose function.
Plot ACF and PACF: Visualize the autocorrelation and partial autocorrelation functions using the plot_acf_pacf function.
Plot Differenced Series: Visualize the differenced series using the plot_differenced_series function.
6. Model Training
Train XGBoost Model: Train an XGBoost model using the XGBoostModel class in xgboost_model.py.
7. Model Evaluation
Evaluate Model: Evaluate the model using the evaluate_model function in xgboost_model.py.
Backtest Model: Perform backtesting using the backtest function in xgboost_model.py.
8. Visualization of Results
Plot Model Performance: Visualize the model performance using the plot_model_performance function.
Plot Predictions: Visualize the predictions using the plot_predictions function.

-------------------

Steps to Push code from VS code to Github.
First authenticate your githib account and integrate with VS code. Click on the source control icon and complete the setup.
1. Click terminal and open new terminal
2. git config --global user.name "Swapnilin"
3. git config --global user.email swapnilforcat@gmail.com
4. git init
5. git add .
6. git commit -m "Your commit message"