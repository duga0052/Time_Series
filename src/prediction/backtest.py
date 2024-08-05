import pandas as pd
import logging
from src.prediction.prediction import make_predictions

logger = logging.getLogger(__name__)

def backtest(data, model, features, start=5031, step=120):
    try:
        all_predictions = []
        for i in range(start, data.shape[0], step):
            train = data.iloc[:i].copy()
            test = data.iloc[i:(i+step)].copy()
            predictions = make_predictions(train, test, features, model)
            all_predictions.append(predictions)
        if not all_predictions:
            raise ValueError("No predictions were made during backtesting.")
        logger.info("Backtesting completed successfully")
        return pd.concat(all_predictions)
    except Exception as e:
        logger.error(f"Failed to backtest: {e}")
        raise