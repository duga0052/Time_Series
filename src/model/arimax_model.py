import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import logging

logger = logging.getLogger(__name__)

class ARIMAXModel:
    def __init__(self, order=(1,1,1)):
        self.order = order
        self.model = None

    def fit(self, endog, exog):
        try:
            self.model = ARIMA(endog, exog=exog, order=self.order)
            self.fitted_model = self.model.fit()
            logger.info("ARIMAX model fitted successfully")
            return self.fitted_model
        except Exception as e:
            logger.error(f"Failed to fit ARIMAX model: {e}")
            raise

    def forecast(self, steps=1, exog=None):
        try:
            forecast = self.fitted_model.get_forecast(steps, exog=exog)
            logger.info("ARIMAX model forecast completed successfully")
            return forecast.predicted_mean, forecast.conf_int()
        except Exception as e:
            logger.error(f"Failed to forecast with ARIMAX model: {e}")
            raise

    def evaluate(self, actual, predicted):
        try:
            mae = mean_absolute_error(actual, predicted)
            logger.info("ARIMAX model evaluation completed successfully")
            return mae
        except Exception as e:
            logger.error(f"Failed to evaluate ARIMAX model: {e}")
            raise

    def summary(self):
        try:
            summary = self.fitted_model.summary()
            logger.info("ARIMAX model summary generated successfully")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate ARIMAX model summary: {e}")
            raise