import logging

logger = logging.getLogger(__name__)

def create_train_test_split(data, test_size=30):
    try:
        train = data.iloc[:-test_size]
        test = data.iloc[-test_size:]
        logger.info("Train-test split created successfully")
        return train, test
    except Exception as e:
        logger.error(f"Failed to create train-test split: {e}")
        raise