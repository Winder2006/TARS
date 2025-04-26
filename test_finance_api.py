#!/usr/bin/env python3
"""
Test script to debug Finance API connection and Bitcoin price retrieval
"""

import os
import json
import logging
import requests
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('finance_api_test')

# Load environment variables
load_dotenv()

def test_finance_api():
    """Test connection to finance APIs and debug Bitcoin price retrieval"""
    
    # Try multiple finance APIs to compare results
    
    # 1. Try CoinGecko API (no key required)
    logger.info("Testing CoinGecko API...")
    try:
        coingecko_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        response = requests.get(coingecko_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            btc_price_coingecko = data.get('bitcoin', {}).get('usd')
            logger.info(f"Bitcoin price from CoinGecko: ${btc_price_coingecko:,}")
        else:
            logger.error(f"CoinGecko API error: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error accessing CoinGecko API: {str(e)}")
    
    # 2. Try Coinbase API (no key required)
    logger.info("\nTesting Coinbase API...")
    try:
        coinbase_url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
        response = requests.get(coinbase_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            btc_price_coinbase = float(data.get('data', {}).get('amount', 0))
            logger.info(f"Bitcoin price from Coinbase: ${btc_price_coinbase:,}")
        else:
            logger.error(f"Coinbase API error: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error accessing Coinbase API: {str(e)}")
    
    # 3. Check the tools implementation in TARS
    logger.info("\nChecking TARS Finance Tool implementation...")
    try:
        from tools import FinanceTool
        finance_tool = FinanceTool()
        
        # Test the finance tool with a direct Bitcoin query
        query = "What is the current price of Bitcoin?"
        logger.info(f"Testing with query: '{query}'")
        
        can_handle = finance_tool.can_handle(query)
        logger.info(f"Can FinanceTool handle this query? {can_handle}")
        
        if can_handle:
            result = finance_tool.execute(query)
            logger.info(f"FinanceTool result: {result}")
        else:
            logger.warning("FinanceTool cannot handle this query")
            
        # Test the direct finance data retrieval method
        logger.info("\nTesting direct price retrieval...")
        try:
            symbol = "BTC-USD"  # Bitcoin to USD
            btc_data = finance_tool.get_crypto_price(symbol)
            logger.info(f"Direct Bitcoin data from FinanceTool: {btc_data}")
            
            if isinstance(btc_data, dict):
                btc_price = btc_data.get('price')
                if btc_price:
                    logger.info(f"Extracted Bitcoin price: ${float(btc_price):,}")
                else:
                    logger.error("Bitcoin price not found in response")
            else:
                logger.error(f"Unexpected response format: {type(btc_data)}")
                
        except Exception as e:
            logger.error(f"Error in direct finance data retrieval: {str(e)}")
    
    except ImportError:
        logger.error("Could not import FinanceTool from tools.py")
    except Exception as e:
        logger.error(f"Error testing TARS FinanceTool: {str(e)}")
    
    # Summary
    logger.info("\n=== Finance API Test Summary ===")
    logger.info("Check the above results to see if there are discrepancies between the APIs")
    logger.info("If TARS is returning incorrect prices, there may be an issue with its API source or parsing logic")

if __name__ == "__main__":
    test_finance_api() 