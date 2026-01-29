#!/usr/bin/env python
"""
ETH ML Historical Training Script
==================================
Trains ETH ML model on 3 months of historical data using SMC strategy.
This script runs ONCE to create an initial trained model.

The bot will then use this model and continuously improve it with real trades.
"""

import os
import sys
import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import datetime

print("=" * 70)
print("🎓 ETH ML TRAINING SIMULATION")
print("=" * 70)
print("This script will train the ETH ML model on 3 months of historical data")
print("using the SMC (Smart Money Concepts) strategy.\n")

# Load environment
load_dotenv()

api_key = os.getenv('API_KEY')
secret = os.getenv('SECRET')

if not api_key or not secret:
    print("❌ Error: API_KEY and SECRET must be set in .env file")
    sys.exit(1)

# Initialize exchange
exchange = ccxt.krakenfutures({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
})

# Parameters
SYMBOL = 'ETH/USD:USD'
TIMEFRAME = '15m'  # Use 15min candles to get more history
LOOKBACK_DAYS = 30  # 1 month (720 candles * 15min = 7.5 days max per request)
SAMPLES_NEEDED = 200
SL_PERCENT = 1.0 / 100  # 1% stop loss
TP_TIER1 = 0.4 / 100  # 0.4% take profit

# ===== SMC FUNCTIONS (copied from main.py) =====

def calculate_rsi(series, period=14):
    """Calculate RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_break_of_structure(df, lookback=20):
    """Detect BOS"""
    if df is None or len(df) < lookback + 5:
        return False
    
    recent = df.tail(lookback)
    swing_highs = []
    swing_lows = []
    
    for i in range(1, len(recent) - 1):
        if recent.iloc[i]['high'] > recent.iloc[i-1]['high'] and recent.iloc[i]['high'] > recent.iloc[i+1]['high']:
            swing_highs.append((i, recent.iloc[i]['high']))
        if recent.iloc[i]['low'] < recent.iloc[i-1]['low'] and recent.iloc[i]['low'] < recent.iloc[i+1]['low']:
            swing_lows.append((i, recent.iloc[i]['low']))
    
    if not swing_highs or not swing_lows:
        return False
    
    last_high = max(swing_highs, key=lambda x: x[1])
    last_low = min(swing_lows, key=lambda x: x[1])
    
    current_high = df['high'].iloc[-1]
    current_low = df['low'].iloc[-1]
    
    if current_high > last_high[1] and current_low < last_low[1]:
        return 'bullish_bos'
    
    if current_low < last_low[1]:
        return 'bearish_bos'
    
    return False

def detect_order_blocks(df, lookback=50):
    """Detect Order Blocks"""
    if df is None or len(df) < lookback:
        return []
    
    recent = df.tail(lookback)
    order_blocks = []
    
    for i in range(1, len(recent) - 2):
        curr = recent.iloc[i]
        next_bar = recent.iloc[i+1]
        
        avg_vol = recent.iloc[max(0, i-10):i]['volume'].mean()
        vol_condition = curr['volume'] > avg_vol * 1.5
        
        body_size = abs(curr['close'] - curr['open'])
        move_condition = body_size > (curr['high'] - curr['low']) * 0.6
        
        if vol_condition and move_condition:
            if curr['close'] > curr['open'] and next_bar['close'] > curr['close']:
                order_blocks.append({'type': 'bullish_ob', 'low': curr['low'], 'high': curr['high']})
            elif curr['close'] < curr['open'] and next_bar['close'] < curr['close']:
                order_blocks.append({'type': 'bearish_ob', 'low': curr['low'], 'high': curr['high']})
    
    return order_blocks

def detect_liquidity_sweeps(df, lookback=30):
    """Detect Liquidity Sweeps"""
    if df is None or len(df) < lookback + 5:
        return False
    
    recent = df.tail(lookback)
    recent_lows = [recent.iloc[i]['low'] for i in range(1, len(recent) - 1)
                   if recent.iloc[i]['low'] < recent.iloc[i-1]['low'] and 
                   recent.iloc[i]['low'] < recent.iloc[i+1]['low']]
    
    recent_highs = [recent.iloc[i]['high'] for i in range(1, len(recent) - 1)
                    if recent.iloc[i]['high'] > recent.iloc[i-1]['high'] and 
                    recent.iloc[i]['high'] > recent.iloc[i+1]['high']]
    
    if not recent_lows and not recent_highs:
        return False
    
    current_low = df['low'].iloc[-1]
    current_high = df['high'].iloc[-1]
    current_close = df['close'].iloc[-1]
    
    if recent_lows and current_low < min(recent_lows) and current_close > min(recent_lows):
        return 'sweep_low'
    
    if recent_highs and current_high > max(recent_highs) and current_close < max(recent_highs):
        return 'sweep_high'
    
    return False

def detect_fair_value_gaps(df, lookback=20):
    """Detect FVGs"""
    if df is None or len(df) < lookback:
        return []
    
    recent = df.tail(lookback)
    fvgs = []
    
    for i in range(1, len(recent) - 1):
        prev_bar = recent.iloc[i-1]
        curr_bar = recent.iloc[i]
        next_bar = recent.iloc[i+1]
        
        if curr_bar['low'] > prev_bar['high'] and next_bar['high'] < curr_bar['low']:
            fvgs.append({'type': 'bullish_fvg', 'top': curr_bar['low'], 'bottom': prev_bar['high']})
        
        if curr_bar['high'] < prev_bar['low'] and next_bar['low'] > curr_bar['high']:
            fvgs.append({'type': 'bearish_fvg', 'top': prev_bar['low'], 'bottom': curr_bar['high']})
    
    return fvgs

def detect_premium_discount_zones(df, lookback=50):
    """Detect Premium/Discount Zones"""
    if df is None or len(df) < lookback:
        return 'neutral'
    
    recent = df.tail(lookback)
    range_high = recent['high'].max()
    range_low = recent['low'].min()
    range_size = range_high - range_low
    
    current_price = df['close'].iloc[-1]
    
    discount_top = range_low + (range_size * 0.3)
    premium_bottom = range_low + (range_size * 0.7)
    
    if current_price <= discount_top:
        return 'in_discount'
    elif current_price >= premium_bottom:
        return 'in_premium'
    else:
        return 'neutral'

def get_buy_signal_eth_smc(df):
    """ETH SMC Buy Signal"""
    if df is None or len(df) < 60:
        return False
    
    bos = detect_break_of_structure(df, lookback=20)
    if bos != 'bullish_bos':
        return False
    
    sweep = detect_liquidity_sweeps(df, lookback=30)
    if sweep != 'sweep_low':
        return False
    
    obs = detect_order_blocks(df, lookback=30)
    current_price = df['close'].iloc[-1]
    near_ob = any(ob['type'] == 'bullish_ob' and 
                  ob['low'] <= current_price <= ob['high'] * 1.01
                  for ob in obs)
    if not near_ob:
        return False
    
    zone = detect_premium_discount_zones(df, lookback=50)
    if zone != 'in_discount':
        return False
    
    fvgs = detect_fair_value_gaps(df, lookback=20)
    if not fvgs:
        return False
    
    volume_spike = df['volume'].iloc[-1] > df['volume'].iloc[-20:].mean() * 1.2
    if not volume_spike:
        return False
    
    return True

def get_sell_signal_eth_smc(df, entry_price):
    """ETH SMC Sell Signal"""
    if df is None or len(df) < 60:
        return False
    
    current_price = df['close'].iloc[-1]
    
    # TP reached
    if current_price >= entry_price * (1 + TP_TIER1):
        return True
    
    # SL hit
    if current_price <= entry_price * (1 - SL_PERCENT):
        return True
    
    # Bearish BOS
    bos = detect_break_of_structure(df, lookback=20)
    if bos == 'bearish_bos':
        return True
    
    # Premium zone
    zone = detect_premium_discount_zones(df, lookback=50)
    if zone == 'in_premium':
        return True
    
    # RSI extreme
    rsi = calculate_rsi(df['close'])
    if rsi.iloc[-1] > 85:
        return True
    
    return False

# ===== SIMULATION =====

print("📊 Step 1: Fetching 3 months of historical data...")
print(f"   Symbol: {SYMBOL}")
print(f"   Timeframe: {TIMEFRAME}")
print(f"   Period: {LOOKBACK_DAYS} days")
print()

try:
    # Kraken limit: 720 candles max per request
    # With 15min candles: 720 * 15min = 10800min = 7.5 days
    # So we can get about 7 days of data in one request
    
    print(f"   Fetching maximum available candles (limit=720)...")
    
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=720)
    
    if not ohlcv:
        print("❌ No data returned")
        sys.exit(1)
    
    all_data = ohlcv
    
    print(f"✅ Fetched {len(all_data)} candles")
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print()

except Exception as e:
    print(f"\n❌ Error fetching data: {e}")
    sys.exit(1)

print("🎮 Step 2: Running SMC strategy simulation...")
print(f"   Generating trading signals on historical data...")
print()

# Simulate trades
trades = []
position = None
trade_num = 0

for i in range(200, len(df) - 100):  # Leave buffer for lookforward
    df_slice = df.iloc[:i+1].copy()
    
    # If no position, look for entry
    if position is None:
        if get_buy_signal_eth_smc(df_slice):
            entry_price = df_slice['close'].iloc[-1]
            position = {
                'entry_idx': i,
                'entry_price': entry_price,
                'entry_time': df_slice['timestamp'].iloc[-1]
            }
            trade_num += 1
            
            if trade_num % 10 == 0:
                print(f"   Trade #{trade_num}: Entry @ ${entry_price:,.2f} ({df_slice['timestamp'].iloc[-1].strftime('%Y-%m-%d')})")
    
    # If in position, look for exit
    else:
        if get_sell_signal_eth_smc(df_slice, position['entry_price']):
            exit_price = df_slice['close'].iloc[-1]
            profit_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
            
            trades.append({
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'profit_pct': profit_pct,
                'duration': i - position['entry_idx'],
                'result': 1 if profit_pct > 0 else 0
            })
            
            position = None

print(f"\n✅ Simulation complete: {len(trades)} trades generated")
print()

# ===== TRAINING =====

print("🧠 Step 3: Training ML model on SMC features...")
print("   (Training on ALL historical data, not just trade signals)")
print()

# Extract SMC features for ALL candles (not just trades)
training_data = []

for i in range(200, len(df) - 100):
    df_slice = df.iloc[:i+1].copy()
    
    # Extract SMC features
    bos = detect_break_of_structure(df_slice, lookback=20)
    bos_val = 1 if bos == 'bullish_bos' else (-1 if bos == 'bearish_bos' else 0)
    
    obs = detect_order_blocks(df_slice, lookback=30)
    bullish_obs = sum(1 for ob in obs if ob['type'] == 'bullish_ob')
    
    sweep = detect_liquidity_sweeps(df_slice, lookback=30)
    sweep_val = 1 if sweep == 'sweep_low' else (-1 if sweep == 'sweep_high' else 0)
    
    fvgs = detect_fair_value_gaps(df_slice, lookback=20)
    fvg_count = len(fvgs)
    
    zone = detect_premium_discount_zones(df_slice, lookback=50)
    zone_val = 1 if zone == 'in_discount' else (-1 if zone == 'in_premium' else 0)
    
    ret_1 = df_slice['close'].pct_change(1).iloc[-1]
    ret_5 = df_slice['close'].pct_change(5).iloc[-1]
    vol_ratio = df_slice['volume'].iloc[-1] / df_slice['volume'].iloc[-20:].mean()
    
    # Label: 1 if price goes up in next 5 candles, 0 otherwise
    if i + 5 < len(df):
        future_price = df.iloc[i + 5]['close']
        current_price = df_slice['close'].iloc[-1]
        label = 1 if future_price > current_price else 0
        
        training_data.append({
            'bos': bos_val,
            'ob_count': bullish_obs,
            'sweep': sweep_val,
            'fvg_count': fvg_count,
            'zone': zone_val,
            'ret_1': ret_1,
            'ret_5': ret_5,
            'vol_ratio': vol_ratio,
            'label': label
        })

train_df = pd.DataFrame(training_data).dropna()

if len(train_df) < 100:
    print(f"❌ Not enough training samples ({len(train_df)}). Need at least 100.")
    sys.exit(1)

X = train_df[['bos', 'ob_count', 'sweep', 'fvg_count', 'zone', 'ret_1', 'ret_5', 'vol_ratio']]
y = train_df['label']

print(f"   Training samples: {len(X)}")
print(f"   Positive samples: {y.sum()} ({y.mean() * 100:.1f}%)")
print(f"   Negative samples: {len(y) - y.sum()} ({(1 - y.mean()) * 100:.1f}%)")
print()

# Train model
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X, y)

print(f"\n✅ Model trained successfully!")
print()

# Test model accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(f"📊 Model Performance:")
print(f"   Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
print()

# ===== SAVE MODEL =====

print("💾 Step 4: Saving model...")

model_path = 'logs/ml_model_eth.joblib'
os.makedirs('logs', exist_ok=True)
joblib.dump(model, model_path)

print(f"✅ Model saved to: {model_path}")
print()

# ===== SUMMARY =====

print("=" * 70)
print("🎉 ETH ML TRAINING COMPLETE!")
print("=" * 70)
print()
print("📈 Summary:")
print(f"   • Simulated trades: {len(trades)}")
print(f"   • Training samples: {len(X)}")
print(f"   • Model accuracy baseline: {y.mean() * 100:.1f}%")
print(f"   • Model file: {model_path}")
print()
print("🚀 Next steps:")
print("   1. Restart the bot: python src/main.py")
print("   2. The bot will automatically load this trained model")
print("   3. ETH will now show ML predictions instead of 'None'")
print("   4. Model will improve over time with real trades")
print()
print("=" * 70)
