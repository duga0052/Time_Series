import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

logger = logging.getLogger(__name__)

def plot_line(data, column, title='Line Plot'):
    try:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data[column])
        plt.title(title)
        plt.xticks(rotation=90)
        plt.show()
        logger.info(f"Line plot for {column} created successfully")
    except Exception as e:
        logger.error(f"Failed to create line plot for {column}: {e}")
        raise

def plot_seasonal_decompose(data, column):
    try:
        # Handle missing values by forward filling
        data_filled = data[column].ffill()
        
        # Ensure the index has a frequency set
        if not isinstance(data.index, pd.DatetimeIndex) or data.index.freq is None:
            data_filled.index = pd.date_range(start=data.index[0], periods=len(data_filled), freq='D')
        
        if len(data_filled) < 730:
            logger.warning("Not enough data for seasonal decomposition (need at least 730 observations)")
            return
        
        decomposed = seasonal_decompose(data_filled, model='additive', period=365)
        trend = decomposed.trend
        seasonal = decomposed.seasonal
        residual = decomposed.resid

        plt.figure(figsize=(12, 8))
        plt.subplot(411)
        plt.plot(data[column], label='Original', color='black')
        plt.legend(loc='upper left')
        plt.subplot(412)
        plt.plot(trend, label='Trend', color='red')
        plt.legend(loc='upper left')
        plt.subplot(413)
        plt.plot(seasonal, label='Seasonal', color='blue')
        plt.legend(loc='upper left')
        plt.subplot(414)
        plt.plot(residual, label='Residual', color='black')
        plt.legend(loc='upper left')
        plt.show()
        logger.info(f"Seasonal decomposition plot for {column} created successfully")
    except Exception as e:
        logger.error(f"Failed to create seasonal decomposition plot for {column}: {e}")
        raise

def plot_acf_pacf(data, column):
    try:
        plt.figure(figsize=(14, 6))
        plt.subplot(121)
        plot_acf(data[column].dropna(), ax=plt.gca())
        plt.subplot(122)
        plot_pacf(data[column].dropna(), ax=plt.gca(), lags=11)
        plt.show()
        logger.info(f"ACF and PACF plots for {column} created successfully")
    except Exception as e:
        logger.error(f"Failed to create ACF and PACF plots for {column}: {e}")
        raise

def plot_differenced_series(data, column):
    try:
        differenced = data[column].diff().dropna()
        plt.figure(figsize=(10, 6))
        plt.plot(differenced)
        plt.title('1st Order Differenced Series')
        plt.xlabel('Date')
        plt.xticks(rotation=30)
        plt.ylabel('Price (USD)')
        plt.show()
        logger.info(f"Differenced series plot for {column} created successfully")
    except Exception as e:
        logger.error(f"Failed to create differenced series plot for {column}: {e}")
        raise

def plot_model_performance(actual, predicted, title='Model Performance'):
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(actual.index, actual, label='Actual')
        plt.plot(actual.index, predicted, label='Predicted', color='orange')
        plt.fill_between(predicted.index, predicted - 1.96 * predicted.std(), predicted + 1.96 * predicted.std(), color='k', alpha=.15)
        plt.title(title)
        plt.legend(loc='lower right')
        plt.xlabel('Date')
        plt.xticks(rotation=30)
        plt.ylabel('Price (USD)')
        plt.show()
        logger.info("Model performance plot created successfully")
    except Exception as e:
        logger.error(f"Failed to create model performance plot: {e}")
        raise

def plot_predictions(actual, predicted, title='Model Performance'):
    try:
        # Convert predicted to Pandas Series if it's a NumPy array
        if isinstance(predicted, np.ndarray):
            predicted = pd.Series(predicted, index=actual.index, name='predictions')
        
        plt.figure(figsize=(10, 6))
        plt.plot(actual.index, actual, label='Actual')
        plt.plot(actual.index, predicted, label='Predicted', color='orange')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.xlabel('Date')
        plt.xticks(rotation=30)
        plt.ylabel('Target')
        plt.show()
        logger.info("Model performance plot created successfully")
    except Exception as e:
        logger.error(f"Failed to create model performance plot: {e}")
        raise