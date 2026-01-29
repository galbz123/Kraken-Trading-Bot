#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
from src.main import *

print('=== Testing ML Functions ===\n')

# Load models
print('1. Loading models...')
for asset in ASSETS.keys():
    model = load_ml_model(asset)
    ml_models[asset] = model
    print(f'   {asset}: {"✓ loaded" if model is not None else "✗ not found"}')

# Test BTC
print('\n2. Testing BTC...')
btc_data = get_historical_data('BTC/USD:USD', limit=100)
if btc_data is not None:
    print(f'   BTC data shape: {btc_data.shape}')
    proba = ml_predict_latest('BTC', btc_data)
    print(f'   BTC ML proba: {proba}')
else:
    print('   ✗ BTC data fetch failed')

# Test ETH  
print('\n3. Testing ETH...')
eth_ltf = get_historical_data('ETH/USD:USD', timeframe='5m', limit=150)
eth_htf = get_historical_data('ETH/USD:USD', timeframe='15m', limit=250)
if eth_ltf is not None and eth_htf is not None:
    print(f'   ETH LTF shape: {eth_ltf.shape}')
    print(f'   ETH HTF shape: {eth_htf.shape}')
    proba = ml_predict_latest('ETH', eth_ltf, df_htf=eth_htf)
    print(f'   ETH ML proba: {proba}')
else:
    print('   ✗ ETH data fetch failed')

print('\n=== Test Complete ===')
