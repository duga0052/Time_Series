import logging

logger = logging.getLogger(__name__)

def select_features(data, features):
    try:
        selected_data = data[features]
        logger.info(f"Features {features} selected successfully")
        return selected_data
    except Exception as e:
        logger.error(f"Failed to select features {features}: {e}")
        raise