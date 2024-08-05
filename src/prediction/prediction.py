import pandas as pd
import logging

logger = logging.getLogger(__name__)

def make_predictions(train, test, features, model):
    try:
        model.fit(train[features], train['Target'])
        predictions = model.predict(test[features])
        predictions_series = pd.Series(predictions, index=test.index, name='predictions')
        logger.info("Predictions made successfully")
        return pd.concat([test['Target'], predictions_series], axis=1)
    except Exception as e:
        logger.error(f"Failed to make predictions: {e}")
        raise