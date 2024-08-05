import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
import logging

logger = logging.getLogger(__name__)

class XGBoostModel:
    def __init__(self, max_depth=3, n_estimators=100, random_state=42):
        self.model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        try:
            self.model.fit(X, y)
            logger.info("XGBoost model fitted successfully")
            return self.model
        except Exception as e:
            logger.error(f"Failed to fit XGBoost model: {e}")
            raise

    def predict(self, X):
        try:
            predictions = self.model.predict(X)
            logger.info("XGBoost model predictions made successfully")
            return predictions
        except Exception as e:
            logger.error(f"Failed to make predictions with XGBoost model: {e}")
            raise

    def evaluate(self, y_true, y_pred):
        try:
            precision = precision_score(y_true, y_pred)
            logger.info("XGBoost model evaluation completed successfully")
            return precision
        except Exception as e:
            logger.error(f"Failed to evaluate XGBoost model: {e}")
            raise

    def backtest(self, data, features, target, start=0, step=5):
        try:
            all_predictions = []
            for i in range(start, data.shape[0], step):
                train = data.iloc[:i].copy()
                test = data.iloc[i:(i+step)].copy()
                if test.empty or train.empty:
                    logger.warning(f"Empty train or test set at index {i}")
                    continue
                self.fit(train[features], train[target])
                predictions = self.predict(test[features])
                predictions_series = pd.Series(predictions, index=test.index, name='predictions')
                combined = pd.concat([test[target], predictions_series], axis=1)
                all_predictions.append(combined)
            
            if not all_predictions:
                raise ValueError("No predictions were made during backtesting.")
            
            logger.info("XGBoost model backtesting completed successfully")
            return pd.concat(all_predictions)
        except Exception as e:
            logger.error(f"Failed to backtest XGBoost model: {e}")
            raise