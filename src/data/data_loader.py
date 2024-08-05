import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise

def convert_date(data, date_column):
    try:
        data[date_column] = pd.to_datetime(data[date_column])
        logger.info(f"Date column {date_column} converted to datetime successfully")
        return data
    except Exception as e:
        logger.error(f"Failed to convert {date_column} to datetime: {e}")
        raise