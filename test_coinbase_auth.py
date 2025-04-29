"""
Simple test script to verify Coinbase API authentication.
"""
import os
import time
import hmac
import hashlib
import base64
import json
import requests
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

API_KEY_FULL = os.getenv('COINBASE_API_KEY', '')
PRIVATE_KEY = os.getenv('COINBASE_PRIVATE_KEY', '')

if '/' in API_KEY_FULL and 'apiKeys/' in API_KEY_FULL:
    API_KEY = API_KEY_FULL.split('apiKeys/')[1]
    logger.info(f"Extracted API key ID: {API_KEY}")
else:
    API_KEY = API_KEY_FULL
    logger.info(f"Using full API key: {API_KEY[:10]}...")

def generate_signature(timestamp, method, request_path, body=""):
    """Generate signature for Coinbase API request."""
    message = timestamp + method + request_path + body
    
    logger.info(f"Signature message: {message}")
    logger.info(f"API Key: {API_KEY[:10]}...")
    
    try:
        private_key = PRIVATE_KEY
        private_key = private_key.replace("-----BEGIN EC PRIVATE KEY-----\\n", "")
        private_key = private_key.replace("\\n-----END EC PRIVATE KEY-----\\n", "")
        private_key = private_key.replace("-----BEGIN EC PRIVATE KEY-----\n", "")
        private_key = private_key.replace("\n-----END EC PRIVATE KEY-----\n", "")
        private_key = private_key.replace("\\n", "")
        
        logger.info(f"Cleaned private key (first 10 chars): {private_key[:10]}...")
        
        try:
            key_bytes = base64.b64decode(private_key)
            logger.info("Successfully decoded private key")
        except Exception as decode_error:
            logger.error(f"Error decoding private key: {decode_error}")
            raise
        
        signature = hmac.new(key_bytes, message.encode('utf-8'), hashlib.sha256).digest()
        encoded_signature = base64.b64encode(signature).decode('utf-8')
        
        logger.info(f"Generated signature (first 10 chars): {encoded_signature[:10]}...")
        
        return encoded_signature
    except Exception as e:
        logger.error(f"Error generating signature: {e}")
        raise

def test_auth():
    """Test authentication with Coinbase API."""
    endpoints = [
        {"endpoint": "/accounts", "url": "https://api.coinbase.com/api/v3/brokerage/accounts"},
        {"endpoint": "/accounts", "url": "https://api.coinbase.com/v2/accounts"},
        {"endpoint": "/accounts", "url": "https://api.exchange.coinbase.com/accounts"}
    ]
    
    timestamp = str(int(time.time()))
    
    for endpoint_info in endpoints:
        endpoint = endpoint_info["endpoint"]
        url = endpoint_info["url"]
        
        logger.info(f"\nTrying endpoint: {url}")
        
        signature = generate_signature(timestamp, "GET", endpoint)
        
        headers = {
            "CB-ACCESS-KEY": API_KEY,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }
        
        if "exchange.coinbase.com" in url:
            headers["CB-ACCESS-PASSPHRASE"] = "your_passphrase"  # This would need to be set if using Coinbase Pro
    
    logger.info("Request Headers:")
    for key, value in headers.items():
        if key == "CB-ACCESS-SIGN":
            logger.info(f"{key}: {value[:10]}...")
        else:
            logger.info(f"{key}: {value}")
    
        try:
            logger.info("Request Headers:")
            for key, value in headers.items():
                if key == "CB-ACCESS-SIGN":
                    logger.info(f"{key}: {value[:10]}...")
                else:
                    logger.info(f"{key}: {value}")
            
            response = requests.get(url, headers=headers)
            
            logger.info(f"Response Status Code: {response.status_code}")
            logger.info(f"Response Headers: {response.headers}")
            
            if response.status_code == 200:
                logger.info("Authentication successful!")
                logger.info(f"Response: {response.json()}")
                return True
            else:
                logger.error(f"Authentication failed: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error making request: {e}")
    
    logger.error("All endpoints failed authentication")
    return False

if __name__ == "__main__":
    logger.info("Testing Coinbase API authentication...")
    test_auth()
