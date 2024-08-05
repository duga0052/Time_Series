import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import logging

logger = logging.getLogger(__name__)

class SARIMAModel:
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None

    def fit(self, data):
        try:
            self.model = SARIMAX(data, order=self.order, seasonal_order=self.seasonal_order)
            self.fitted_model = self.model.fit(disp=False)
            logger.info("SARIMA model fitted successfully")
            return self.fitted_model
        except Exception as e:
            logger.error(f"Failed to fit SARIMA model: {e}")
            raise

    def forecast(self, steps=1):
        try:
            forecast = self.fitted_model.get_forecast(steps)
            logger.info("SARIMA model forecast completed successfully")
            return forecast.predicted_mean, forecast.conf_int()
        except Exception as e:
            logger.error(f"Failed to forecast with SARIMA model: {e}")
            raise

    def evaluate(self, actual, predicted):
        try:
            mae = mean_absolute_error(actual, predicted)
            logger.info("SARIMA model evaluation completed successfully")
            return mae
        except Exception as e:
            logger.error(f"Failed to evaluate SARIMA model: {e}")
            raise

    def summary(self):
        try:
            summary = self.fitted_model.summary()
            logger.info("SARIMA model summary generated successfully")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate SARIMA model summary: {e}")
            raise