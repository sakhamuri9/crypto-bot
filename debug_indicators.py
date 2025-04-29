"""
Debug script to diagnose issues with indicators in mock data.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from mock_coinbase_client import MockCoinbaseClient
from indicators import add_indicators
from strategy import TradingStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def main():
    """Main debug function."""
    client = MockCoinbaseClient()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    symbol = 'BTC-USD'
    interval = '1h'

    logger.info('Fetching mock data...')
    df = client.get_historical_candles(symbol, interval, start_date, end_date)

    logger.info(f'Data shape: {df.shape}')
    logger.info(f'Columns: {df.columns.tolist()}')

    if 'ema9' in df.columns:
        logger.info('ema9 is present in the dataframe')
        logger.info(f'First few values of ema9: {df["ema9"].head()}')
    else:
        logger.info('ema9 is NOT present in the dataframe')
        
        logger.info('Trying to add indicators manually...')
        try:
            df_with_indicators = add_indicators(df)
            logger.info(f'Columns after adding indicators: {df_with_indicators.columns.tolist()}')
            if 'ema9' in df_with_indicators.columns:
                logger.info('ema9 is now present after manually adding indicators')
                logger.info(f'First few values of ema9: {df_with_indicators["ema9"].head()}')
            else:
                logger.info('ema9 is still NOT present after manually adding indicators')
        except Exception as e:
            logger.error(f'Error adding indicators: {e}')
    
    logger.info('Trying to generate signals...')
    try:
        strategy = TradingStrategy()
        df_with_signals = strategy.generate_signals(df)
        logger.info('Successfully generated signals')
        logger.info(f'Columns after generating signals: {df_with_signals.columns.tolist()}')
    except Exception as e:
        logger.error(f'Error generating signals: {e}')
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
