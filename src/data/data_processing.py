import logging

logger = logging.getLogger(__name__)

def preprocess_data(data):
    try:
        if 'AAPL' not in data.columns:
            raise ValueError("Column 'AAPL' not found in data")
        
        data['Next_day'] = data['AAPL'].shift(-1)
        data['Target'] = (data['Next_day'] > data['AAPL']).astype(int)
        logger.info("Data preprocessing completed successfully")
        return data
    except Exception as e:
        logger.error(f"Failed to preprocess data: {e}")
        raise