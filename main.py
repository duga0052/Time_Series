import logging
import os
from src.data.data_loader import load_data, convert_date
from src.data.data_processing import preprocess_data
from src.feature.feature_selection import select_features
from src.feature.train_test_split import create_train_test_split
from src.model.xgboost_model import XGBoostModel
from src.visualization.plots import plot_line, plot_seasonal_decompose, plot_acf_pacf, plot_differenced_series, plot_model_performance, plot_predictions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the log file exists
log_file_exists = os.path.exists('app.log')
if not log_file_exists:
    with open('app.log', 'w') as f:
        f.write('Log file created.\n')

# Configure logging
def setup_logging():
    logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

setup_logging()

def main():
    try:
        # Load and preprocess data
        data_path = 'AAPL.csv'
        data = load_data(data_path)
        data = convert_date(data, 'Date')
        
        logger.info(f"Data columns: {data.columns}")
        logger.info(f"Data shape: {data.shape}")

        data = preprocess_data(data)
        logger.info("Data loaded and preprocessed successfully.")

        # Feature selection and train-test split
        features = ['AAPL', 'TXN']
        target = 'Target'
        
        # Ensure all required columns are present
        for col in features + [target]:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in dataset")

        # Adjust train-test split to ensure sufficient training data
        train_size = int(0.8 * len(data))
        train, test = data.iloc[:train_size], data.iloc[train_size:]
        logger.info("Train-test split created successfully.")
        logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")

        # Plot line
        plot_line(data, 'AAPL', title='Stock Price Over Time')

       
        # Plot seasonal decomposition if enough data
        if len(data) >= 730:
            plot_seasonal_decompose(data, 'AAPL')
        else:
            logger.warning("Not enough data for seasonal decomposition (need at least 730 observations)")

        # Plot ACF and PACF
        plot_acf_pacf(data, 'AAPL')

        # Plot differenced series
        plot_differenced_series(data, 'AAPL')

        # Train and evaluate XGBoost model
        xgb_model = XGBoostModel()
        xgb_model.fit(train[features], train[target])
        predictions = xgb_model.predict(test[features])
        logger.info("XGBoost model trained and predictions made successfully.")

        # Backtest
        backtest_results = xgb_model.backtest(data, features, target)
        logger.info("XGBoost model backtested successfully.")
        logger.info(f"Backtest results shape: {backtest_results.shape}")

        # Visualize results
        plot_predictions(test[target], predictions, title='Model Performance')
        logger.info("Results visualized successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()