import os
import time
import ccxt
import pandas as pd
import numpy as np
import datetime
import json
import threading
from dotenv import load_dotenv
from telegram import Bot
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import requests
import asyncio
import sys
from oci_secret import load_trading_secrets_into_env

load_trading_secrets_into_env()

# Unbuffer output
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf8', buffering=1)

# Load environment variables
load_dotenv()

# Kraken Futures API setup
api_key = os.getenv('API_KEY') or os.getenv('KRAKEN_API_KEY')
secret = os.getenv('SECRET') or os.getenv('KRAKEN_API_SECRET')

telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
cryptopanic_token = os.getenv('CRYPTOPANIC_API_KEY')

if not api_key or not secret:
    raise ValueError("API credentials are missing. Set API_KEY/SECRET or KRAKEN_API_KEY/KRAKEN_API_SECRET")

# Core trading parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_OVERBOUGHT = 80
STOCH_OVERSOLD = 20
ATR_PERIOD = 14

max_trades_total = 14
tp_tiers = {
    'tp1': {'offset': 0.4, 'percent': 40},
    'tp2': {'offset': 0.8, 'percent': 30},
    'tp3': {'offset': 1.5, 'percent': 30}
}

US_MARKET_OPEN_HOUR_UTC = 14
US_MARKET_OPEN_MINUTE_UTC = 30
US_MARKET_FILTER_RANGE = 15  # minutes

ml_min_samples = 80
ml_lookahead = 3

news_cache = None
news_cache_time = None
news_cache_ttl = 300
news_cooldown_duration = 1800  # 30 minutes
news_cooldown_end = 0

# Asset configuration
ASSETS = {
    'BTC': {
        'symbol': 'BTC/USD:USD',
        'amount': 0.0018,
        'sl_percent': 1.5,
        'ml_threshold': 0.55,
        'ml_model_path': 'logs/ml_model_btc.joblib'
    },
    'ETH': {
        'symbol': 'ETH/USD:USD',
        'amount': 0.045,
        'sl_percent': 1.5,
        'ltf_timeframe': '5m',
        'htf_timeframe': '15m',
        'ml_threshold_base': 0.58,
        'ml_threshold_strong': 0.52,
        'ml_model_path': 'logs/ml_model_eth.joblib',
        'daily_cap': 7,
        'cooldown_minutes': 15,
        'last_trade_time': None
    }
}

ML_FEATURE_COLUMNS = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d', 'atr', 'ret_1', 'ret_5']

# State containers
positions = {'BTC': None, 'ETH': None}
trade_count = 0
trade_count_eth = 0
ml_models = {'BTC': None, 'ETH': None}
ml_last_train = {}
last_reset_date = None

# Initialize exchange
exchange = ccxt.krakenfutures({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
})
exchange.load_markets()

# Telegram bot
bot = None
if telegram_token and telegram_chat_id:
    try:
        bot = Bot(token=telegram_token)
    except Exception as e:
        print(f"Telegram init failed: {e}")

def now_str():
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Asia/Jerusalem")
        return datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def us_open_local_str():
    try:
        from zoneinfo import ZoneInfo
        utc_today = datetime.datetime.utcnow().replace(hour=US_MARKET_OPEN_HOUR_UTC, minute=US_MARKET_OPEN_MINUTE_UTC, second=0, microsecond=0)
        local = utc_today.replace(tzinfo=datetime.timezone.utc).astimezone(ZoneInfo("Asia/Jerusalem"))
        return local.strftime("%H:%M")
    except Exception:
        return f"{US_MARKET_OPEN_HOUR_UTC:02d}:{US_MARKET_OPEN_MINUTE_UTC:02d}"

def send_telegram_message(message):
    if not bot:
        print(f"⚠️ [TELEGRAM NOT INITIALIZED] Message not sent: {message}")
        return
    try:
        # Use asyncio to handle the async send_message call
        asyncio.run(bot.send_message(chat_id=telegram_chat_id, text=message))
    except Exception as e:
        print(f"Telegram send error: {e}")

def _append_log(event_type, asset, data=None):
    entry = {
        'ts': datetime.datetime.utcnow().isoformat(),
        'type': event_type,
        'asset': asset
    }
    if data:
        entry.update(data)
    os.makedirs('logs', exist_ok=True)
    with open('logs/trades.json', 'a') as f:
        f.write(json.dumps(entry) + "\n")

def calculate_rsi(close_prices, period=RSI_PERIOD):
    """Calculate RSI"""
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(close_prices, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """Calculate MACD and signal/histogram"""
    ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calculate_stochastic(high, low, close, k_period=STOCH_K_PERIOD, d_period=STOCH_D_PERIOD):
    """Calculate Stochastic Oscillator values."""
    highest_high = high.rolling(window=k_period).max()
    lowest_low = low.rolling(window=k_period).min()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_k = stoch_k.rolling(window=1).mean()
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d

def calculate_atr(high, low, close, period=ATR_PERIOD):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_ema(series, period):
    """Calculate EMA"""
    return series.ewm(span=period, adjust=False).mean()

def find_pivot_highs(df, window=3):
    """
    Find pivot highs (local maxima) using rolling window
    Returns: Series of pivot high prices (NaN where no pivot)
    """
    highs = df['high'].rolling(window=window*2+1, center=True).apply(
        lambda x: x[window] if x[window] == max(x) else np.nan, raw=True
    )
    return highs

def find_pivot_lows(df, window=3):
    """
    Find pivot lows (local minima) using rolling window
    Returns: Series of pivot low prices (NaN where no pivot)
    """
    lows = df['low'].rolling(window=window*2+1, center=True).apply(
        lambda x: x[window] if x[window] == min(x) else np.nan, raw=True
    )
    return lows

def calculate_spread(symbol):
    """
    Calculate bid-ask spread if available
    Returns: spread value or None
    """
    try:
        ticker = exchange.fetch_ticker(symbol)
        if 'bid' in ticker and 'ask' in ticker and ticker['bid'] and ticker['ask']:
            spread = ticker['ask'] - ticker['bid']
            return spread, ticker['bid'], ticker['ask']
        return None, None, None
    except:
        return None, None, None

# ===== DATA FETCH =====
def get_historical_data(symbol, timeframe='1m', limit=100):
    """Fetch OHLCV data from Kraken Futures with retries and richer error context"""
    for attempt in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            status = None
            body = None
            if hasattr(e, 'response') and e.response is not None:
                status = getattr(e.response, 'status_code', None)
                body = getattr(e.response, 'text', None)
            msg = f"Error fetching data for {symbol} (attempt {attempt + 1}/3): {type(e).__name__} {e}"
            if status is not None:
                msg += f" | status={status}"
            if body:
                msg += f" | body={body[:200]}"
            print(msg)
            time.sleep(2 * (attempt + 1))  # simple backoff
    return None

# ===== ML FUNCTIONS =====
def load_ml_model(asset):
    """Load ML model from disk"""
    model_path = ASSETS[asset]['ml_model_path']
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            model.set_output(transform="pandas")
            print(f"ML: loaded saved model for {asset} ✅")
            return model
        except Exception as e:
            print(f"ML: error loading {asset} model: {e}")
    return None


def build_ml_feature_matrix(asset, df):
    """Construct feature matrix for ML models"""
    if df is None or len(df) < 20:
        return None
    
    rsi = calculate_rsi(df['close'])
    macd, macd_signal, macd_hist = calculate_macd(df['close'])
    stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
    atr = calculate_atr(df['high'], df['low'], df['close'])
    ret_1 = df['close'].pct_change(1)
    ret_5 = df['close'].pct_change(5)
    
    X = pd.concat([
        rsi, macd, macd_signal, macd_hist, stoch_k, stoch_d, atr, ret_1, ret_5
    ], axis=1)
    X.columns = ML_FEATURE_COLUMNS
    X = X.dropna()
    return X


def get_latest_ml_features(asset, df):
    """Return latest ML feature row as dict"""
    X = build_ml_feature_matrix(asset, df)
    if X is None or len(X) == 0:
        return None
    return X.iloc[-1].to_dict()


def train_ml_model(asset, df):
    """Train logistic regression model for asset (indicator feature set)"""
    global ml_last_train
    
    if df is None or len(df) < ml_min_samples:
        return False
    
    try:
        X = build_ml_feature_matrix(asset, df)
        if X is None or len(X) < ml_min_samples:
            return False
        
        # Create labels aligned with X's index
        y = (df['close'].shift(-ml_lookahead) > df['close']).astype(int)
        y = y[X.index]
        
        # Ensure same length
        if len(y) != len(X):
            print(f"ML: label-feature mismatch for {asset} (len(y)={len(y)}, len(X)={len(X)})")
            return False
        
        if len(y[y == 1]) == 0:
            return False
        
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.set_output(transform="pandas")
        model.fit(X, y)
        
        model_path = ASSETS[asset]['ml_model_path']
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        
        model.set_output(transform="pandas")
        ml_last_train[asset] = datetime.datetime.utcnow()
        print(f"ML: model trained on {len(X)} samples (indicators) at {ml_last_train[asset]} for {asset}")
        return True
    except Exception as e:
        print(f"ML train error for {asset}: {e}")
        return False


def ml_predict_latest(asset, df_ltf, df_htf=None):
    """Get ML probability for latest bar"""
    global ml_models
    
    if asset not in ml_models or ml_models[asset] is None:
        return None
    
    X = build_ml_feature_matrix(asset, df_ltf)
    if X is None or len(X) == 0:
        return None
    
    try:
        latest_features = X.iloc[-1:]
        proba = ml_models[asset].predict_proba(latest_features)[0][1]
        return proba
    except Exception as e:
        print(f"ML predict error for {asset}: {e}")
        import traceback
        traceback.print_exc()
        return None


def append_outcome_sample(asset, features, label, profit_pct, exit_price, entry_price, reason):
    """Persist trade outcome sample for online updates"""
    if features is None:
        return
    record = {
        'ts': datetime.datetime.utcnow().isoformat(),
        'asset': asset,
        'label': int(label),
        'profit_pct': profit_pct,
        'exit_price': exit_price,
        'entry_price': entry_price,
        'reason': reason,
        'features': {k: float(features.get(k)) for k in ML_FEATURE_COLUMNS if features.get(k) is not None}
    }
    path = f"logs/ml_outcomes_{asset.lower()}.jsonl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(record) + "\n")


def train_ml_from_outcomes(asset, min_samples=8):
    """Retrain model using accumulated trade outcomes"""
    global ml_models, ml_last_train
    path = f"logs/ml_outcomes_{asset.lower()}.jsonl"
    if not os.path.exists(path):
        return False
    
    rows = []
    labels = []
    try:
        with open(path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    feats = entry.get('features', {})
                    if all(k in feats for k in ML_FEATURE_COLUMNS):
                        rows.append({k: feats[k] for k in ML_FEATURE_COLUMNS})
                        labels.append(int(entry.get('label', 0)))
                except Exception:
                    continue
        if len(rows) < min_samples:
            return False
        X = pd.DataFrame(rows)[ML_FEATURE_COLUMNS]
        y = pd.Series(labels)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.set_output(transform="pandas")
        model.fit(X, y)
        model_path = ASSETS[asset]['ml_model_path']
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        model.set_output(transform="pandas")
        ml_models[asset] = model
        ml_last_train[asset] = datetime.datetime.utcnow()
        print(f"ML: outcome-based update for {asset} using {len(rows)} samples")
        return True
    except Exception as e:
        print(f"ML outcome-train error ({asset}): {e}")
        return False


def maybe_reset_daily_counters():
    """Reset trade counters at start of local day"""
    global trade_count, trade_count_eth, last_reset_date
    try:
        from zoneinfo import ZoneInfo
        now_local = datetime.datetime.now(ZoneInfo("Asia/Jerusalem"))
    except Exception:
        now_local = datetime.datetime.utcnow()
    today = now_local.date().isoformat()
    if last_reset_date != today:
        trade_count = 0
        trade_count_eth = 0
        last_reset_date = today
        print(f"[DAILY RESET] Counters reset for {today} (BTC/ETH/total)")


def fetch_live_positions_map():
    """Fetch live positions once and map by symbol"""
    try:
        all_positions = exchange.fetch_positions()
        mapped = {}
        for pos in all_positions:
            symbol = pos.get('symbol')
            if symbol:
                mapped[symbol] = pos
        return mapped
    except Exception as e:
        print(f"[SYNC] fetch_positions error: {e}")
        return {}


def fetch_order_status(order_id, symbol):
    try:
        order = exchange.fetch_order(order_id, symbol)
        return order.get('status'), order
    except Exception:
        return None, None


def cancel_take_profits(asset, tp_ids=None):
    """Cancel TP orders for asset (ONLY the specific TP order IDs provided)"""
    if not tp_ids:
        return  # אם אין tp_ids, אל תבטל כלום (מונע ביטול סטופ לוס בטעות)
    symbol = ASSETS[asset]['symbol']
    try:
        open_orders = exchange.fetch_open_orders(symbol)
        for order in open_orders:
            if order['side'] == 'sell' and order['id'] in tp_ids:
                try:
                    exchange.cancel_order(order['id'], symbol)
                    print(f"[TP CANCEL] {asset} canceled TP {order['id']}")
                except Exception as e_cancel:
                    print(f"[TP CANCEL] {asset} error canceling {order['id']}: {e_cancel}")
    except Exception as e:
        print(f"[TP CANCEL] {asset} open order fetch failed: {e}")


def handle_position_close(asset, exit_price=None, reason_hint=None):
    """Handle local position close: telegram, logs, ML, cleanup"""
    global positions
    pos = positions.get(asset)
    if pos is None:
        return
    symbol = ASSETS[asset]['symbol']
    entry_price = pos.get('entry_price') or exit_price or 0
    if exit_price is None:
        try:
            ticker = exchange.fetch_ticker(symbol)
            exit_price = ticker.get('last') or ticker.get('close')
        except Exception as e:
            print(f"[CLOSE] {asset} price fetch failed: {e}")
    if exit_price is None:
        exit_price = entry_price
    profit = (exit_price - entry_price)
    profit_pct = (profit / entry_price) * 100 if entry_price else 0
    reason = 'CLOSED'
    sl_status, _ = (None, None)
    if pos.get('sl_id'):
        sl_status, _ = fetch_order_status(pos['sl_id'], symbol)
    tp_hit = False
    if pos.get('tp_ids'):
        for tp_id in pos['tp_ids']:
            status, _ = fetch_order_status(tp_id, symbol)
            if status == 'closed':
                tp_hit = True
                break
    if sl_status == 'closed':
        reason = 'SL'
    elif tp_hit:
        reason = 'TP'
    elif reason_hint:
        reason = reason_hint
    cancel_take_profits(asset, pos.get('tp_ids'))
    positions[asset] = None
    msg = f"{asset} position closed ({reason}) @ {exit_price:.2f}, PnL {profit_pct:.2f}%"
    print(msg)
    if bot:
        send_telegram_message(msg)
    _append_log('close', asset, {
        'exit_price': exit_price,
        'entry_price': entry_price,
        'profit_pct': profit_pct,
        'reason': reason
    })
    label = 1 if profit_pct > 0 else 0
    append_outcome_sample(asset, pos.get('ml_features'), label, profit_pct, exit_price, entry_price, reason)
    train_ml_from_outcomes(asset)


def sync_position_state(asset, live_positions_map, latest_price=None):
    """Compare local vs live positions; detect closes and recover stray positions"""
    config = ASSETS[asset]
    symbol = config['symbol']
    live_pos = live_positions_map.get(symbol)
    local_pos = positions.get(asset)
    live_contracts = live_pos.get('contracts') if live_pos else 0
    
    if local_pos and (live_pos is None or live_contracts in (None, 0)):
        handle_position_close(asset, latest_price)
        return
    if not local_pos and live_pos and live_contracts not in (None, 0):
        entry_price = live_pos.get('entryPrice')
        positions[asset] = {
            'side': live_pos.get('side', 'long'),
            'amount': live_contracts,
            'entry_price': entry_price,
            'entry_time': datetime.datetime.utcnow().isoformat(),
            'tp_ids': [],
            'sl_id': None,
            'ml_features': None
        }
        print(f"[SYNC] Recovered live {asset} position: {live_contracts} @ {entry_price}")
        return

# ===== NEWS FUNCTIONS =====
def fetch_news_crypto(token_symbol='bitcoin'):
    """Fetch news from CryptoPanic"""
    global news_cache, news_cache_time
    
    if news_cache is not None and news_cache_time is not None:
        if time.time() - news_cache_time < news_cache_ttl:
            return news_cache
    
    if not cryptopanic_token:
        return []
    
    try:
        url = f"https://cryptopanic.com/api/v1/posts/"
        params = {
            'auth_token': cryptopanic_token,
            'filter': 'trending',
            'currencies': token_symbol,
            'kind': 'news'
        }
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            news_cache = data.get('results', [])
            news_cache_time = time.time()
            return news_cache
    except Exception as e:
        print(f"News fetch error: {e}")
    
    return []

def news_score(news_list, token_symbol='BTC'):
    """Score news sentiment"""
    negative_words = ['hack', 'exploit', 'lawsuit', 'ban', 'outage', 'bankrupt', 'downtime', 'breach', 'probe', 'investigation', 'shutdown']
    score = 0
    now = time.time()
    cutoff_time = now - 3600
    
    for news in news_list:
        published_at = datetime.datetime.fromisoformat(news.get('published_at', '').replace('Z', '+00:00')).timestamp()
        if published_at < cutoff_time:
            continue
        
        title = (news.get('title', '') + ' ' + news.get('body', '')).lower()
        
        for neg_word in negative_words:
            if neg_word in title:
                score -= 1
    
    return score

def news_should_block():
    """Check if news-based block is active"""
    global news_cooldown_end
    
    if time.time() < news_cooldown_end:
        return True
    
    news_list = fetch_news_crypto('bitcoin')
    score = news_score(news_list, 'BTC')
    
    if score < 0:
        news_cooldown_end = time.time() + news_cooldown_duration
        print(f"News: negative sentiment detected (score={score}), blocking buys for 30 min")
        if bot:
            send_telegram_message(f"⚠️ CryptoPanic negative news detected (score={score}), blocking buys for 30 min")
        return True
    
    return False

def get_news_sentiment():
    """Get current news sentiment score"""
    news_list = fetch_news_crypto('bitcoin')
    return news_score(news_list, 'BTC')

# ===== US MARKET FILTER =====
def is_us_market_open_window():
    """Check if we're in US market open window"""
    now = datetime.datetime.utcnow()
    hour = now.hour
    minute = now.minute
    
    target_time = now.replace(hour=US_MARKET_OPEN_HOUR_UTC, minute=US_MARKET_OPEN_MINUTE_UTC, second=0, microsecond=0)
    time_diff = abs((now - target_time).total_seconds()) / 60
    
    if time_diff <= US_MARKET_FILTER_RANGE:
        return True
    return False

# ===== POSITION RECOVERY =====
def recover_state():
    """Recover position and trade count by checking live positions on exchange"""
    global positions, trade_count, ml_models
    
    try:
        # Load ML models
        for asset in ASSETS.keys():
            model = load_ml_model(asset)
            ml_models[asset] = model
        
        # Check LIVE positions from exchange (most reliable)
        print("Recovering state from exchange...")
        all_positions = exchange.fetch_positions()
        
        for pos in all_positions:
            symbol = pos.get('symbol')
            contracts = pos.get('contracts')
            
            # Find which asset this position belongs to
            for asset, asset_config in ASSETS.items():
                if asset_config['symbol'] == symbol and contracts is not None and contracts != 0:
                    entry_price = pos.get('entryPrice')
                    side = pos.get('side', 'long')
                    
                    positions[asset] = {
                        'side': side,
                        'amount': contracts,
                        'entry_price': entry_price,
                        'entry_time': None,  # Unknown from position data
                        'tp_ids': [],  # Will be recovered from open orders
                        'sl_id': None
                    }
                    print(f"  Recovered {asset} position: {contracts} @ ${entry_price:,.2f}")
        
        # Check for open TP/SL orders
        for asset, asset_config in ASSETS.items():
            if positions[asset] is not None:
                try:
                    open_orders = exchange.fetch_open_orders(asset_config['symbol'])
                    for order in open_orders:
                        if order['side'] == 'sell':
                            # Could be TP or SL
                            if positions[asset]['tp_ids'] is None:
                                positions[asset]['tp_ids'] = []
                            positions[asset]['tp_ids'].append(order['id'])
                except Exception as e_orders:
                    print(f"  Could not fetch open orders for {asset}: {e_orders}")
        
        # Count trades from log
        if os.path.exists('logs/trades.json'):
            trade_count = 0
            with open('logs/trades.json', 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get('type') == 'buy':
                            trade_count += 1
                    except:
                        pass
        
        print(f"State recovered: BTC={positions['BTC'] is not None}, ETH={positions['ETH'] is not None}, trade_count={trade_count}")
        maybe_reset_daily_counters()
    except Exception as e:
        print(f"State recovery error: {e}")
        import traceback
        traceback.print_exc()


def has_open_position_live(symbol):
    """
    Check if there's an ACTIVE position for a symbol by querying the exchange.
    Returns True if position exists, False otherwise.
    This is a LIVE check to prevent duplicate entries!
    """
    try:
        # Method 1: Fetch ALL positions and find our symbol
        all_positions = exchange.fetch_positions()
        
        for pos in all_positions:
            if pos.get('symbol') == symbol:
                contracts = pos.get('contracts')
                entry_price = pos.get('entryPrice')
                side = pos.get('side')
                
                # Position exists if contracts is not None and != 0
                if contracts is not None and contracts != 0:
                    print(f"[LIVE CHECK] ⚠️  ACTIVE POSITION DETECTED!")
                    print(f"   Symbol: {symbol}")
                    print(f"   Contracts: {contracts}")
                    print(f"   Entry Price: ${entry_price:,.2f}" if entry_price else "   Entry Price: N/A")
                    print(f"   Side: {side}")
                    return True
        
        # Method 2: Check for recent filled buy orders in last 15 minutes
        recent_orders = exchange.fetch_orders(symbol, limit=30)
        import time
        fifteen_min_ago = (time.time() - 900) * 1000  # 15 minutes in milliseconds
        
        recent_buys = []
        for order in recent_orders:
            if (order['status'] == 'closed' and 
                order['side'] == 'buy' and 
                order['timestamp'] > fifteen_min_ago):
                recent_buys.append(order)
        
        if recent_buys:
            print(f"[LIVE CHECK] ⚠️  RECENT BUY ORDERS FOUND: {len(recent_buys)} in last 15min")
            for buy in recent_buys:
                print(f"   - {buy['amount']} @ ${buy.get('average', buy.get('price', 0)):,.2f} at {buy['datetime']}")
            return True
        
        print(f"[LIVE CHECK] ✅ No open position for {symbol}")
        return False
        
    except Exception as e:
        print(f"[LIVE CHECK ERROR] {symbol}: {e}")
        import traceback
        traceback.print_exc()
        # On error, return TRUE (conservative - better to skip than duplicate)
        print(f"[LIVE CHECK] ⚠️  Error occurred - assuming position EXISTS to be safe")
        return True

# ===== TRADING LOGIC =====

def get_higher_timeframe_bias_eth(symbol='ETH/USD:USD', hour_lookback=4):
    """
    Multi-timeframe analysis for ETH:
    Use higher timeframe (4h) to confirm direction bias.
    Returns: 'bullish', 'bearish', or 'neutral'
    
    Institutional traders use HTF for macro bias, LTF for entries.
    """
    try:
        # Fetch 4-hour candles
        ohlcv_4h = exchange.fetch_ohlcv(symbol, '4h', limit=50)
        df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        if df_4h is None or len(df_4h) < 20:
            return 'neutral'
        
        # Check 4H structure
        bos_4h = detect_break_of_structure(df_4h)
        
        # Get 4H premium/discount
        zone_4h = detect_premium_discount_zones(df_4h, lookback=50)
        
        # 4H RSI for momentum
        rsi_4h = calculate_rsi(df_4h['close'])
        
        # Bullish Bias: Bullish BOS on 4H + in discount + RSI rising
        if bos_4h == 'bullish_bos' and zone_4h == 'in_discount' and rsi_4h.iloc[-1] > 40:
            return 'bullish'
        
        # Bearish Bias: Bearish BOS on 4H + in premium + RSI falling
        if bos_4h == 'bearish_bos' and zone_4h == 'in_premium' and rsi_4h.iloc[-1] < 60:
            return 'bearish'
        
        return 'neutral'
    
    except Exception as e:
        print(f"HTF bias error: {e}")
        return 'neutral'

def get_buy_signal_btc(df):
    """BTC buy signal: RSI <30 crossed from >=30 AND Stoch K <20"""
    if df is None or len(df) < RSI_PERIOD + 5:
        return False
    
    rsi = calculate_rsi(df['close'])
    stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
    
    rsi_crossed = (rsi.iloc[-2] >= RSI_OVERSOLD) and (rsi.iloc[-1] < RSI_OVERSOLD)
    stoch_ready = stoch_k.iloc[-1] < STOCH_OVERSOLD
    
    return rsi_crossed and stoch_ready

def get_sell_signal_btc(df):
    """BTC sell signal"""
    if df is None or len(df) < MACD_SLOW + 5:
        return False
    
    rsi = calculate_rsi(df['close'])
    macd, macd_signal, _ = calculate_macd(df['close'])
    stoch_k, _ = calculate_stochastic(df['high'], df['low'], df['close'])
    
    rsi_crossed = (rsi.iloc[-2] <= RSI_OVERBOUGHT) and (rsi.iloc[-1] > RSI_OVERBOUGHT)
    macd_crossed = (macd.iloc[-2] >= macd_signal.iloc[-2]) and (macd.iloc[-1] < macd_signal.iloc[-1])
    stoch_ready = stoch_k.iloc[-1] > STOCH_OVERBOUGHT
    
    return rsi_crossed and macd_crossed and stoch_ready

# ===== SMC (SMART MONEY CONCEPTS) FUNCTIONS FOR ETH (v2.0 - Improved Logic) =====

def compute_atr(df, period=14):
    """Compute Average True Range"""
    if df is None or len(df) < period:
        return None
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr = np.zeros(len(df))
    tr[0] = high[0] - low[0]
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    atr = pd.Series(tr).rolling(period).mean()
    return atr.iloc[-1] if len(atr) > 0 else None

def find_last_valid_pivot_high(df, lookback=50, window=3, prominence_window=10):
    """Find the last meaningful pivot high (not from last 1-2 candles)"""
    if df is None or len(df) < lookback + window * 2:
        return None
    
    recent = df.tail(lookback)
    pivot_highs = find_pivot_highs(recent, window=window)
    valid_pivots = []
    
    for idx, val in enumerate(pivot_highs.items()):
        if pd.notna(val[1]):
            # Check prominence: is this pivot higher than surrounding candles
            center_idx = idx
            start_idx = max(0, center_idx - prominence_window)
            end_idx = min(len(recent), center_idx + prominence_window + 1)
            region = recent.iloc[start_idx:end_idx]['high'].max()
            
            # Valid if it's the max in the region and not in last 2 candles
            if val[1] >= region * 0.99 and idx < len(recent) - 2:
                valid_pivots.append((idx, val[1]))
    
    return valid_pivots[-1][1] if valid_pivots else None

def find_last_valid_pivot_low(df, lookback=50, window=3, prominence_window=10):
    """Find the last meaningful pivot low"""
    if df is None or len(df) < lookback + window * 2:
        return None
    
    recent = df.tail(lookback)
    pivot_lows = find_pivot_lows(recent, window=window)
    valid_pivots = []
    
    for idx, val in enumerate(pivot_lows.items()):
        if pd.notna(val[1]):
            center_idx = idx
            start_idx = max(0, center_idx - prominence_window)
            end_idx = min(len(recent), center_idx + prominence_window + 1)
            region = recent.iloc[start_idx:end_idx]['low'].min()
            
            if val[1] <= region * 1.01 and idx < len(recent) - 2:
                valid_pivots.append((idx, val[1]))
    
    return valid_pivots[-1][1] if valid_pivots else None

def detect_fvg_zones_v2(df, lookback=30):
    """
    Detect Fair Value Gaps using 3-candle definition:
    Bullish FVG: high[i-1] < low[i+1]
    Returns list of FVG zones with metadata
    """
    if df is None or len(df) < 5:
        return []
    
    recent = df.tail(lookback)
    fvgs = []
    
    for i in range(1, len(recent) - 1):
        prev_high = recent.iloc[i-1]['high']
        curr_low = recent.iloc[i]['low']
        next_low = recent.iloc[i+1]['low']
        
        # Bullish FVG: prev_high < curr_low (gap up)
        if prev_high < curr_low:
            gap_size = curr_low - prev_high
            if gap_size > recent.iloc[i]['close'] * 0.001:  # At least 0.1%
                fvgs.append({
                    'type': 'bullish_fvg',
                    'bottom': prev_high,
                    'top': curr_low,
                    'mid': (prev_high + curr_low) / 2,
                    'size': gap_size,
                    'candle_idx': i
                })
    
    return fvgs[-5:] if len(fvgs) > 5 else fvgs

def detect_bullish_ob_v2(df, lookback=50, atr=None):
    """
    Detect bullish order block based on displacement + confirmation
    Returns: (ob_low, ob_high) or None
    Bullish OB = last bearish candle before a displacement leg
    """
    if df is None or len(df) < lookback:
        return None
    
    if atr is None:
        atr = compute_atr(df, period=14)
    if atr is None:
        return None
    
    recent = df.tail(lookback)
    disp_atr_mult = 1.2
    min_displacement_range = disp_atr_mult * atr
    
    # Find displacement candle (big range + good body)
    for i in range(len(recent) - 3):
        candle = recent.iloc[i]
        candle_range = candle['high'] - candle['low']
        body_size = abs(candle['close'] - candle['open'])
        
        if candle_range >= min_displacement_range and body_size >= 0.60 * candle_range:
            # Check if this was preceded by opposite candle (OB candidate)
            if i > 0:
                ob_candle = recent.iloc[i-1]
                # Bullish displacement -> OB should be bearish (close < open)
                if candle['close'] > candle['open']:  # Bullish displacement
                    if ob_candle['close'] < ob_candle['open']:  # Bearish OB
                        ob_low = ob_candle['low']
                        ob_high = max(ob_candle['open'], ob_candle['close'])
                        return (ob_low, ob_high)
    
    return None

def check_sweep_quality(df, sweep_candle_idx, pivot_low, atr):
    """Check if a sweep has good wick dominance"""
    if sweep_candle_idx < 0 or sweep_candle_idx >= len(df):
        return False
    
    candle = df.iloc[sweep_candle_idx]
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    range_size = candle['high'] - candle['low']
    
    if range_size == 0:
        return False
    
    lower_wick_ratio = lower_wick / range_size
    return lower_wick_ratio >= 0.35

# ===== OLD FUNCTIONS (kept for compatibility) =====

def detect_break_of_structure_v2(df, lookback=20, window=3):
    """
    Improved BOS detection using pivot points
    - Bullish BOS: Current close breaks above recent pivot high
    - Bearish BOS: Current close breaks below recent pivot low
    Returns: 'bullish_bos', 'bearish_bos', or False
    """
    if df is None or len(df) < lookback + window * 2:
        return False
    
    # Find pivot highs and lows
    pivot_highs = find_pivot_highs(df, window=window)
    pivot_lows = find_pivot_lows(df, window=window)
    
    # Get last valid pivots
    recent_pivot_highs = pivot_highs.dropna().tail(lookback)
    recent_pivot_lows = pivot_lows.dropna().tail(lookback)
    
    if len(recent_pivot_highs) == 0 or len(recent_pivot_lows) == 0:
        return False
    
    last_pivot_high = recent_pivot_highs.iloc[-1] if len(recent_pivot_highs) > 0 else None
    last_pivot_low = recent_pivot_lows.iloc[-1] if len(recent_pivot_lows) > 0 else None
    
    current_close = df['close'].iloc[-1]
    current_high = df['high'].iloc[-1]
    current_low = df['low'].iloc[-1]
    
    # Bullish BOS: Price breaks above pivot high
    if last_pivot_high and current_high > last_pivot_high:
        return 'bullish_bos'
    
    # Bearish BOS: Price breaks below pivot low
    if last_pivot_low and current_low < last_pivot_low:
        return 'bearish_bos'
    
    return False

def detect_break_of_structure(df, lookback=20):
    """
    Detect Break of Structure (BOS):
    - Bullish BOS: Price breaks above previous swing high (mark reversal)
    - Bearish BOS: Price breaks below previous swing low
    Returns: 'bullish_bos', 'bearish_bos', or False
    """
    if df is None or len(df) < lookback + 5:
        return False
    
    recent = df.tail(lookback)
    
    # Find swing highs and lows in lookback period
    swing_highs = []
    swing_lows = []
    
    for i in range(1, len(recent) - 1):
        if recent.iloc[i]['high'] > recent.iloc[i-1]['high'] and recent.iloc[i]['high'] > recent.iloc[i+1]['high']:
            swing_highs.append((i, recent.iloc[i]['high']))
        if recent.iloc[i]['low'] < recent.iloc[i-1]['low'] and recent.iloc[i]['low'] < recent.iloc[i+1]['low']:
            swing_lows.append((i, recent.iloc[i]['low']))
    
    if not swing_highs or not swing_lows:
        return False
    
    # Get the last significant swing high and low
    last_high = max(swing_highs, key=lambda x: x[1])
    last_low = min(swing_lows, key=lambda x: x[1])
    
    current_close = df['close'].iloc[-1]
    current_high = df['high'].iloc[-1]
    current_low = df['low'].iloc[-1]
    
    # Bullish BOS: Current close > Last swing high
    if current_high > last_high[1] and current_low < last_low[1]:
        return 'bullish_bos'
    
    # Bearish BOS: Current close < Last swing low
    if current_low < last_low[1]:
        return 'bearish_bos'
    
    return False

def detect_order_blocks(df, lookback=50):
    """
    Detect Order Blocks:
    Institutional areas where large orders sit. Typically the candle before
    a strong impulsive move. Looking for:
    - High volume
    - Sharp directional move
    - Price returning to test
    
    Returns: List of order block zones with price levels
    """
    if df is None or len(df) < lookback:
        return []
    
    recent = df.tail(lookback)
    order_blocks = []
    
    for i in range(1, len(recent) - 2):
        curr = recent.iloc[i]
        next_bar = recent.iloc[i+1]
        
        # Conditions for order block:
        # 1. High volume (1.5x average)
        # 2. Strong directional move (close >> open or vice versa)
        # 3. Next bar continues strongly
        
        avg_vol = recent.iloc[max(0, i-10):i]['volume'].mean()
        vol_condition = curr['volume'] > avg_vol * 1.5
        
        body_size = abs(curr['close'] - curr['open'])
        move_condition = body_size > (curr['high'] - curr['low']) * 0.6
        
        if vol_condition and move_condition:
            # Bullish OB (last bar before up move)
            if curr['close'] > curr['open'] and next_bar['close'] > curr['close']:
                order_blocks.append({
                    'type': 'bullish_ob',
                    'low': curr['low'],
                    'high': curr['high'],
                    'mid': (curr['low'] + curr['high']) / 2,
                    'index': i
                })
            
            # Bearish OB (last bar before down move)
            elif curr['close'] < curr['open'] and next_bar['close'] < curr['close']:
                order_blocks.append({
                    'type': 'bearish_ob',
                    'low': curr['low'],
                    'high': curr['high'],
                    'mid': (curr['low'] + curr['high']) / 2,
                    'index': i
                })
    
    return order_blocks[-3:] if len(order_blocks) > 3 else order_blocks

def check_order_block_proximity(df, order_blocks, tolerance=0.02):
    """
    Check if current price is near an order block zone
    tolerance: 2% proximity threshold
    Returns: True if price is in or near a bullish OB
    """
    if not order_blocks:
        return False
    
    current_price = df['close'].iloc[-1]
    
    for ob in order_blocks:
        if ob['type'] == 'bullish_ob':
            # Check if price is within OB zone or slightly above/below
            ob_low_range = ob['low'] * (1 - tolerance)
            ob_high_range = ob['high'] * (1 + tolerance)
            
            if ob_low_range <= current_price <= ob_high_range:
                return True
    
    return False

def detect_liquidity_sweeps(df, lookback=30, window=3):
    """
    Improved Liquidity Sweeps using pivot points:
    - Sweep Low: Price breaks below pivot low then closes back above it (1-3 candles)
    - Sweep High: Price breaks above pivot high then closes back below it
    Returns: 'sweep_low', 'sweep_high', or False
    """
    if df is None or len(df) < lookback + window * 2:
        return False
    
    # Find recent pivot lows and highs
    pivot_lows = find_pivot_lows(df, window=window)
    pivot_highs = find_pivot_highs(df, window=window)
    
    recent_pivot_lows = pivot_lows.dropna().tail(10)
    recent_pivot_highs = pivot_highs.dropna().tail(10)
    
    if len(recent_pivot_lows) == 0 or len(recent_pivot_highs) == 0:
        return False
    
    last_pivot_low = recent_pivot_lows.iloc[-1]
    last_pivot_high = recent_pivot_highs.iloc[-1]
    
    # Check last 3 candles for sweep pattern
    for i in range(1, min(4, len(df))):
        candle = df.iloc[-i]
        current_close = df['close'].iloc[-1]
        
        # Sweep Low: Low touched below pivot, then close recovered above
        if candle['low'] < last_pivot_low * 0.999 and current_close > last_pivot_low:
            return 'sweep_low'
        
        # Sweep High: High touched above pivot, then close fell below
        if candle['high'] > last_pivot_high * 1.001 and current_close < last_pivot_high:
            return 'sweep_high'
    
    return False

def detect_fair_value_gaps(df, lookback=30):
    """
    Detect Fair Value Gaps (FVG) / Imbalances:
    Area where price moved rapidly, leaving an imbalance.
    Price often returns to fill these gaps.
    
    Returns: List of FVG zones
    """
    if df is None or len(df) < lookback:
        return []
    
    recent = df.tail(lookback)
    fvgs = []
    
    for i in range(1, len(recent) - 1):
        prev = recent.iloc[i-1]
        curr = recent.iloc[i]
        next_bar = recent.iloc[i+1]
        
        # Bullish FVG: Gap up (prev high < curr low)
        if prev['high'] < curr['low']:
            gap_size = curr['low'] - prev['high']
            # FVG should be at least 0.1% of price
            if gap_size > prev['close'] * 0.001:
                fvgs.append({
                    'type': 'bullish_fvg',
                    'top': curr['low'],
                    'bottom': prev['high'],
                    'mid': (curr['low'] + prev['high']) / 2,
                    'size': gap_size
                })
        
        # Bearish FVG: Gap down (curr high < prev low)
        if curr['high'] < prev['low']:
            gap_size = prev['low'] - curr['high']
            if gap_size > prev['close'] * 0.001:
                fvgs.append({
                    'type': 'bearish_fvg',
                    'top': prev['low'],
                    'bottom': curr['high'],
                    'mid': (prev['low'] + curr['high']) / 2,
                    'size': gap_size
                })
    
    return fvgs[-3:] if len(fvgs) > 3 else fvgs

def detect_premium_discount_zones(df, lookback=100):
    """
    Identify Premium (high) vs Discount (low) zones:
    - Discount: Buy when price is in lower 30% of recent range
    - Premium: Sell when price is in upper 30% of recent range
    
    Returns: 'in_discount', 'in_premium', or 'neutral'
    """
    if df is None or len(df) < lookback:
        return 'neutral'
    
    recent = df.tail(lookback)
    
    high = recent['high'].max()
    low = recent['low'].min()
    range_size = high - low
    
    current = df['close'].iloc[-1]
    
    discount_threshold = low + (range_size * 0.30)
    premium_threshold = high - (range_size * 0.30)
    
    if current < discount_threshold:
        return 'in_discount'
    elif current > premium_threshold:
        return 'in_premium'
    else:
        return 'neutral'

def calculate_smc_score_eth(df_htf, df_ltf):
    """
    Calculate SMC Score (0-6 points) based on 6 conditions (IMPROVED v2.0):
    A) HTF Bullish Bias (15m): EMA50 > EMA200 = 1 pt
    B) BOS Bullish on LTF (5m): close > pivot_high + 0.10*ATR = 1 pt
    C) Liquidity Sweep on LTF: sweep with wick quality = 1 pt
    D) Order Block Proximity on LTF: within 0.25*ATR of OB = 1 pt
    E) Discount Zone on LTF: price <= 50% of swing range = 1 pt
    F) FVG Present & Touched/Forming on LTF = 1 pt
    
    Returns: (score, details_dict)
    """
    score = 0
    details = {}
    
    atr = compute_atr(df_ltf, period=14) if df_ltf is not None else None
    
    # (A) HTF Bullish Bias - 15m timeframe (IMPROVED: only EMA50 > EMA200)
    if df_htf is not None and len(df_htf) >= 200:
        ema50_htf = calculate_ema(df_htf['close'], 50)
        ema200_htf = calculate_ema(df_htf['close'], 200)
        
        if ema50_htf.iloc[-1] > ema200_htf.iloc[-1]:
            score += 1
            details['htf_bias'] = True
        else:
            details['htf_bias'] = False
    else:
        details['htf_bias'] = False
    
    # (B) BOS Bullish on LTF (IMPROVED: ATR-based, pivot-based)
    bos = False
    if df_ltf is not None and atr is not None:
        pivot_high = find_last_valid_pivot_high(df_ltf, lookback=50, window=3)
        if pivot_high is not None:
            current_close = df_ltf['close'].iloc[-1]
            bos_threshold = pivot_high + 0.10 * atr
            if current_close > bos_threshold:
                bos = True
                score += 1
    details['bos'] = bos
    
    # (C) Liquidity Sweep (IMPROVED: pivot-based, wick quality)
    sweep = False
    if df_ltf is not None and atr is not None and len(df_ltf) >= 30:
        pivot_low = find_last_valid_pivot_low(df_ltf, lookback=50, window=3)
        if pivot_low is not None:
            recent = df_ltf.tail(30)
            min_low_idx = recent['low'].idxmin()
            min_low_value = recent.loc[min_low_idx, 'low']
            current_close = df_ltf['close'].iloc[-1]
            
            sweep_threshold = pivot_low - 0.05 * atr
            if min_low_value < sweep_threshold and current_close > pivot_low:
                # Check wick quality on the sweep candle
                sweep_candle_idx = len(df_ltf) - (len(recent) - recent.index.get_loc(min_low_idx))
                if check_sweep_quality(df_ltf, sweep_candle_idx, pivot_low, atr):
                    sweep = True
                    score += 1
    details['sweep'] = sweep
    
    # (D) Order Block Proximity (IMPROVED: displacement-based, ATR proximity)
    ob_proximity = False
    if df_ltf is not None and atr is not None:
        ob = detect_bullish_ob_v2(df_ltf, lookback=50, atr=atr)
        if ob is not None:
            ob_low, ob_high = ob
            current_price = df_ltf['close'].iloc[-1]
            
            # Check if within OB zone or within 0.25*ATR of it
            ob_distance = 0
            if current_price < ob_low:
                ob_distance = ob_low - current_price
            elif current_price > ob_high:
                ob_distance = current_price - ob_high
            
            if ob_distance <= 0.25 * atr:
                ob_proximity = True
                score += 1
    details['ob_proximity'] = ob_proximity
    
    # (E) Discount Zone (IMPROVED: swing-based range)
    discount_zone = False
    if df_ltf is not None and len(df_ltf) >= 100:
        pivot_high = find_last_valid_pivot_high(df_ltf, lookback=100, window=3)
        pivot_low = find_last_valid_pivot_low(df_ltf, lookback=100, window=3)
        
        if pivot_high is not None and pivot_low is not None and pivot_high > pivot_low:
            swing_range = pivot_high - pivot_low
            discount_level = pivot_low + 0.50 * swing_range  # 50% of swing
            current_close = df_ltf['close'].iloc[-1]
            
            if current_close <= discount_level:
                discount_zone = True
                score += 1
    details['discount_zone'] = discount_zone
    
    # (F) FVG Touched or Forming (IMPROVED: 3-candle definition, fresh FVG)
    fvg_touched = False
    if df_ltf is not None and atr is not None:
        fvgs = detect_fvg_zones_v2(df_ltf, lookback=30)
        current_price = df_ltf['close'].iloc[-1]
        
        for fvg in fvgs:
            if fvg['type'] == 'bullish_fvg':
                # Check if price touches or is inside FVG (with small buffer)
                fvg_bottom_buffered = fvg['bottom'] - 0.05 * atr
                fvg_top_buffered = fvg['top'] + 0.05 * atr
                
                if fvg_bottom_buffered <= current_price <= fvg_top_buffered:
                    fvg_touched = True
                    break
                
                # Also check if fresh FVG (formed in last 3 candles) and price above bottom
                if fvg['candle_idx'] >= len(df_ltf) - 3 and current_price > fvg['bottom']:
                    fvg_touched = True
                    break
        
        if fvg_touched:
            score += 1
    details['fvg_touched'] = fvg_touched
    
    return score, details

def check_entry_trigger(df):
    """
    Entry Trigger: Bullish confirmation candle
    - Close > Open (bullish candle)
    - Close > High of previous candle (engulfing-ish)
    
    Returns: True if trigger confirmed
    """
    if df is None or len(df) < 2:
        return False
    
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    # Bullish candle
    is_bullish = current['close'] > current['open']
    
    # Close above previous high
    breaks_previous_high = current['close'] > previous['high']
    
    return is_bullish and breaks_previous_high

def get_buy_signal_eth_smc(df_htf, df_ltf):
    """
    NEW Advanced SMC Strategy for ETH with Score-Based Entry:
    
    Score Calculation (0-6):
    - HTF Bullish Bias (1pt)
    - BOS Bullish (1pt)
    - Liquidity Sweep (1pt)
    - Order Block Proximity (1pt)
    - Discount Zone (1pt)
    - FVG Touched (1pt)
    
    Entry Requirements:
    1. SMC Score >= 4
    2. Entry Trigger (bullish confirmation candle)
    3. Hard Gates passed (checked in main loop)
    
    Returns: (should_enter: bool, smc_score: int, details: dict)
    """
    if df_htf is None or df_ltf is None:
        return False, 0, {}
    
    if len(df_htf) < 200 or len(df_ltf) < 100:
        return False, 0, {}
    
    # Calculate SMC Score
    smc_score, details = calculate_smc_score_eth(df_htf, df_ltf)
    
    # Minimum score threshold
    if smc_score < 4:
        return False, smc_score, details
    
    # Entry Trigger
    trigger_confirmed = check_entry_trigger(df_ltf)
    if not trigger_confirmed:
        return False, smc_score, details
    
    details['entry_trigger'] = trigger_confirmed
    
    return True, smc_score, details

def get_buy_signal_eth_smc_legacy(df):
    """
    OLD SMC Strategy - Keep for reference
    [DEPRECATED - Use get_buy_signal_eth_smc instead]
    """
    if df is None or len(df) < 100:
        return False
    
    # Check Structure
    bos = detect_break_of_structure(df)
    if bos != 'bullish_bos':
        return False
    
    # Check for Liquidity Sweep (institutional setup)
    sweep = detect_liquidity_sweeps(df)
    if sweep != 'sweep_low':
        return False
    
    # Check Order Blocks for support
    obs = detect_order_blocks(df)
    if not obs:
        return False
    
    bullish_obs = [ob for ob in obs if ob['type'] == 'bullish_ob']
    if not bullish_obs:
        return False
    
    # Check current price vs OB
    current_price = df['close'].iloc[-1]
    nearest_ob = min(bullish_obs, key=lambda x: abs(x['mid'] - current_price))
    
    # Price should be near or in OB range
    if current_price < nearest_ob['low'] * 0.98 or current_price > nearest_ob['high'] * 1.02:
        return False
    
    # Premium/Discount Zone check
    zone = detect_premium_discount_zones(df)
    if zone != 'in_discount':
        return False
    
    # FVG should be present (to provide take profit target)
    fvgs = detect_fair_value_gaps(df)
    if not fvgs:
        return False
    
    # Volume confirmation (institutional activity)
    avg_vol = df['volume'].iloc[-20:-1].mean()
    if df['volume'].iloc[-1] < avg_vol * 0.8:
        return False
    
    print(f"[ETH SMC] 🎯 PREMIUM SETUP DETECTED - BOS✓ Sweep✓ OB✓ Discount✓ FVG✓ Volume✓")
    return True

def detect_choch_confirmation(df, lookback=30):
    """
    CHOCH = Change of Character
    Identifies when smart money changes direction:
    - Previous impulsive direction changes to retracement direction
    - Signals potential reversal/continuation setup
    
    Returns: 'bullish_choch', 'bearish_choch', or False
    """
    if df is None or len(df) < lookback:
        return False
    
    recent = df.tail(lookback)
    
    # Find the direction of most recent impulsive move
    recent_high = recent['high'].rolling(5).max()
    recent_low = recent['low'].rolling(5).min()
    
    # Impulsive direction: are we making higher highs?
    making_hh = recent['high'].iloc[-1] > recent['high'].iloc[-10]
    making_ll = recent['low'].iloc[-1] < recent['low'].iloc[-10]
    
    # Check if direction is changing
    last_5_opens = recent['open'].tail(5).tolist()
    last_5_closes = recent['close'].tail(5).tolist()
    
    # Count direction reversals
    reversals = 0
    for i in range(1, len(last_5_closes)):
        prev_direction = last_5_closes[i-1] - last_5_opens[i-1]
        curr_direction = last_5_closes[i] - last_5_opens[i]
        if (prev_direction > 0 and curr_direction < 0) or (prev_direction < 0 and curr_direction > 0):
            reversals += 1
    
    if reversals >= 2 and making_hh:
        return 'bullish_choch'
    elif reversals >= 2 and making_ll:
        return 'bearish_choch'
    
    return False

def get_sell_signal_eth(df):
    """
    Advanced SMC Exit Strategy for ETH:
    
    Exit Conditions (institutional-grade):
    1. Price reaches FVG (takes profit at imbalance)
    2. Bearish BOS (structure breaks, trend reverses)
    3. Reached Premium zone with confirmation
    4. Breaker block failed (previous OB becomes resistance)
    
    Returns: True if exit criteria met
    """
    if df is None or len(df) < 50:
        return False
    
    current_price = df['close'].iloc[-1]
    
    # Exit Condition 1: Price reached FVG (take profit target)
    fvgs = detect_fair_value_gaps(df)
    for fvg in fvgs:
        if fvg['type'] == 'bullish_fvg':
            # If we're in a bullish FVG, exit at top
            if current_price > fvg['top'] * 0.99:
                print(f"[ETH SMC] 📈 EXIT: Price reached FVG level {fvg['top']:.2f}")
                return True
    
    # Exit Condition 2: Bearish BOS (structure reversal)
    bos = detect_break_of_structure(df)
    if bos == 'bearish_bos':
        print(f"[ETH SMC] 📉 EXIT: Bearish BOS detected")
        return True
    
    # Exit Condition 3: In Premium zone with volume confirmation
    zone = detect_premium_discount_zones(df)
    if zone == 'in_premium':
        avg_vol = df['volume'].iloc[-20:-1].mean()
        if df['volume'].iloc[-1] > avg_vol * 1.5:
            print(f"[ETH SMC] 📊 EXIT: Premium zone + volume confirmation")
            return True
    
    # Exit Condition 4: Breaker block (failed OB as resistance)
    obs = detect_order_blocks(df)
    bearish_obs = [ob for ob in obs if ob['type'] == 'bearish_ob']
    
    for ob in bearish_obs:
        # If price touches breaker block (previous OB turned resistance)
        if current_price > ob['high'] * 1.005 and current_price < ob['high'] * 1.02:
            # Check for reversal candle
            if df['close'].iloc[-1] < df['open'].iloc[-1]:
                print(f"[ETH SMC] 🚫 EXIT: Breaker block rejection at {ob['high']:.2f}")
                return True
    
    # Fallback: Check RSI overbought as extra confirmation
    rsi = calculate_rsi(df['close'])
    if rsi.iloc[-1] > 85:
        print(f"[ETH SMC] ⚠️  EXIT: Extreme overbought (RSI {rsi.iloc[-1]:.1f})")
        return True
    
    return False

# ===== MARKET EXECUTION =====
def place_order(symbol, side, amount, order_type='market', params=None):
    """Place order"""
    try:
        if params is None:
            params = {}
        order = exchange.create_order(symbol, order_type, side, amount, params=params)
        return order
    except Exception as e:
        print(f"[ERROR] Order failed for {symbol}: {e}")
        return None

def place_stop_loss(symbol, entry_price, amount, sl_percent):
    """Place stop loss order"""
    # Calculate SL price with more precision for ETH
    if 'ETH' in symbol:
        sl_price = round(entry_price * (1 - sl_percent/100), 1)  # 1 decimal for ETH
    else:
        sl_price = round(entry_price * (1 - sl_percent/100), 0)  # 0 decimals for BTC
    
    try:
        # For Kraken Futures: use stop market order with reduceOnly
        order = exchange.create_order(
            symbol,
            'stop',
            'sell',
            amount,
            None,
            params={
                'stopPrice': sl_price,
                'triggerSignal': 'mark',
                'reduceOnly': True
            }
        )
        sl_id = order.get('id')
        print(f"[SL ✓] {symbol}: {amount} @ {sl_price} (Entry: {entry_price}, SL%: {sl_percent}%) | ID: {sl_id}")
        return sl_id
    except Exception as e:
        error_msg = str(e)
        print(f"[SL ✗] {symbol}: Entry={entry_price}, SL_Price={sl_price}, Amount={amount}, SL%={sl_percent}%")
        print(f"[SL ✗] Error details: {error_msg}")
        if 'invalidPrice' in error_msg:
            print(f"[SL ✗] Kraken rejected price {sl_price}. Distance from entry: {abs(entry_price - sl_price):.2f} ({sl_percent}%)")
        return None

def place_take_profit_orders(symbol, entry_price, amount, tier_configs):
    """Place take profit orders"""
    tp_ids = []
    
    for tier_name, tier_config in tier_configs.items():
        tp_price = round(entry_price * (1 + tier_config['offset']/100), 2)
        tp_amount = round(amount * tier_config['percent'] / 100, 6)
        
        try:
            order = exchange.create_order(
                symbol,
                'limit',
                'sell',
                tp_amount,
                tp_price,
                params={'postOnly': True}
            )
            tp_ids.append(order.get('id'))
            print(f"TP placed for {symbol}: {tp_amount} @ {tp_price}")
        except Exception as e:
            print(f"TP order error ({symbol}): {e}")
    
    return tp_ids

# ===== HARD GATES FOR ETH =====
def check_eth_hard_gates(df_ltf, eth_config):
    """
    Check all HARD GATES for ETH entry
    If any gate fails, entry is blocked.
    
    Gates:
    1. Operational: No position open, trades < daily cap, cooldown passed
    2. Volatility/Execution: ATR check, spread check
    3. News/Sentiment: News sentiment check with SMC override
    
    Returns: (passed: bool, reason: str)
    """
    global positions, trade_count, trade_count_eth
    diag = {
        'atr_ratio': None,
        'spread_ratio': None
    }
    
    # === 2.1 OPERATIONAL GATES ===
    
    # Position check
    if positions['ETH'] is not None:
        return False, "Position already open"
    
    # Daily cap check (ETH-specific)
    if trade_count_eth >= eth_config['daily_cap']:
        return False, f"ETH daily cap reached ({trade_count_eth}/{eth_config['daily_cap']})"
    
    # Total trades check
    if trade_count >= max_trades_total:
        return False, f"Total daily cap reached ({trade_count}/{max_trades_total})"
    
    # Cooldown check
    if eth_config['last_trade_time'] is not None:
        time_since_last = (datetime.datetime.utcnow() - eth_config['last_trade_time']).total_seconds() / 60
        if time_since_last < eth_config['cooldown_minutes']:
            return False, f"Cooldown active ({time_since_last:.1f}/{eth_config['cooldown_minutes']} min)"
    
    # === 2.2 VOLATILITY/EXECUTION GATES ===
    
    if df_ltf is not None and len(df_ltf) >= 14:
        # ATR check
        atr = calculate_atr(df_ltf['high'], df_ltf['low'], df_ltf['close'], period=14)
        current_price = df_ltf['close'].iloc[-1]
        atr_ratio = atr.iloc[-1] / current_price
        
        if atr_ratio < 0.0015:  # 0.15% minimum volatility
            return False, f"ATR too low ({atr_ratio*100:.3f}% < 0.15%)"
        
        # Spread check (if available)
        spread, bid, ask = calculate_spread(eth_config['symbol'])
        if spread is not None and current_price > 0:
            spread_ratio = spread / current_price
            if spread_ratio > 0.0005:  # 0.05% max spread
                return False, f"Spread too high ({spread_ratio*100:.3f}% > 0.05%)"
    
    # === 2.3 NEWS/SENTIMENT GATE ===
    # This will be checked in main loop with SMC override logic
    
    return True, "All gates passed"

# ===== MAIN LOOP =====
def main_loop():
    global positions, trade_count, trade_count_eth, ml_models, ml_last_train
    
    print("\n=== Kraken Futures Dual-Asset Bot (BTC + ETH) ===")
    print(f"Max trades total: {max_trades_total}")
    print(f"US market filter: 14:30 UTC ({us_open_local_str()} Israel) ±{US_MARKET_FILTER_RANGE}min")
    print(f"BTC: {ASSETS['BTC']['amount']} @ SL {ASSETS['BTC']['sl_percent']}%")
    print(f"ETH: {ASSETS['ETH']['amount']} @ SL {ASSETS['ETH']['sl_percent']}%")
    print("=" * 50 + "\n")
    
    recover_state()
    
    while True:
        try:
            maybe_reset_daily_counters()
            live_positions_map = fetch_live_positions_map()
            
            # ===== BTC STRATEGY =====
            btc_config = ASSETS['BTC']
            btc_data = get_historical_data(btc_config['symbol'], limit=100)
            
            if btc_data is not None:
                if 'BTC' not in ml_last_train or (datetime.datetime.utcnow() - ml_last_train['BTC']).total_seconds() > 1800:
                    train_ml_model('BTC', btc_data)
                
                ml_proba_btc = ml_predict_latest('BTC', btc_data)
                btc_price = btc_data['close'].iloc[-1]
                
                if ml_proba_btc is not None:
                    proba_str = f"{ml_proba_btc:.3f}"
                else:
                    proba_str = "None"

                btc_latest_features = get_latest_ml_features('BTC', btc_data)
                
                # === BTC LOG (Simple indicator-based strategy) ===
                btc_status = '✓ Ready' if positions['BTC'] is None else '✗ Position Open'
                print(f"{now_str()} [BTC] Price: {btc_price:.2f}, ML: {proba_str}, Trades: {trade_count}/{max_trades_total}, Status: {btc_status}")
                sync_position_state('BTC', live_positions_map, btc_price)
                
                if (positions['BTC'] is None and 
                    trade_count < max_trades_total and 
                    get_buy_signal_btc(btc_data) and 
                    ml_proba_btc and ml_proba_btc >= btc_config['ml_threshold'] and
                    not news_should_block() and
                    not is_us_market_open_window()):
                    
                    # CRITICAL: Triple check - local state + live exchange check
                    if positions['BTC'] is not None:
                        print(f"[BTC] ⚠️  Position already exists (local), skipping buy")
                        continue
                    
                    # LIVE CHECK from exchange to prevent duplicate positions
                    if has_open_position_live(btc_config['symbol']):
                        print(f"[BTC] ⚠️  LIVE CHECK: Position exists on exchange, skipping buy")
                        continue
                    
                    print(f"[BTC] BUY signal triggered: pos={positions['BTC']}, trades={trade_count}/{max_trades_total}")
                    try:
                        # Mark position as pending to prevent duplicate entries
                        positions['BTC'] = 'PENDING'
                        
                        buy_order = place_order(btc_config['symbol'], 'buy', btc_config['amount'])
                        if buy_order is None:
                            print("[BTC] Buy order failed, skipping")
                            positions['BTC'] = None  # Reset on failure
                        else:
                            entry_price = buy_order.get('average')
                            if entry_price is None:
                                entry_price = btc_price
                            
                            sl_id = place_stop_loss(btc_config['symbol'], entry_price, btc_config['amount'], btc_config['sl_percent'])
                            tp_ids = place_take_profit_orders(btc_config['symbol'], entry_price, btc_config['amount'], tp_tiers)
                            
                            # Only update position after successful order
                            positions['BTC'] = {
                                'side': 'long',
                                'amount': btc_config['amount'],
                                'entry_price': entry_price,
                                'entry_time': datetime.datetime.utcnow().isoformat(),
                                'tp_ids': tp_ids,
                                'sl_id': sl_id,
                                'ml_features': btc_latest_features
                            }
                            trade_count += 1
                            print(f"[BTC] Position confirmed: entry={entry_price}, count={trade_count}")
                            
                            msg = f"🚀 BTC BUY: {btc_config['amount']} @ {entry_price:.0f}, SL@{entry_price * (1-btc_config['sl_percent']/100):.0f}, Trade #{trade_count}/6"
                            print(msg)
                            if bot:
                                send_telegram_message(msg)
                            
                            _append_log('buy', 'BTC', {'price': entry_price, 'amount': btc_config['amount']})
                    except Exception as e:
                        print(f"[ERROR] BTC buy logic: {e}")
                        _append_log('error', 'BTC', {'error': str(e)})
                
                if positions['BTC'] is not None and get_sell_signal_btc(btc_data):
                    try:
                        sell_order = place_order(btc_config['symbol'], 'sell', positions['BTC']['amount'])
                        if sell_order is None:
                            print("[BTC] Sell order failed, skipping")
                        else:
                            exit_price = sell_order.get('average')
                            if exit_price is None:
                                exit_price = btc_price
                            handle_position_close('BTC', exit_price, reason_hint='SELL_SIGNAL')
                    except Exception as e:
                        print(f"[ERROR] BTC sell logic: {e}")
                        _append_log('error', 'BTC', {'error': str(e)})
            else:
                print(f"{now_str()} [BTC] Data fetch failed")
                sync_position_state('BTC', live_positions_map, None)
            
            # ===== ETH STRATEGY (UPGRADED SMC) =====
            eth_config = ASSETS['ETH']
            
            # Fetch both timeframes for ETH
            eth_htf = get_historical_data(eth_config['symbol'], timeframe=eth_config['htf_timeframe'], limit=250)
            eth_ltf = get_historical_data(eth_config['symbol'], timeframe=eth_config['ltf_timeframe'], limit=150)
            
            if eth_htf is not None and eth_ltf is not None:
                try:
                    # Train ML model periodically (use LTF data)
                    if 'ETH' not in ml_last_train or (datetime.datetime.utcnow() - ml_last_train['ETH']).total_seconds() > 1800:
                        train_ml_model('ETH', eth_ltf)
                    
                    # Get ML prediction (with HTF context for ETH)
                    ml_proba_eth = ml_predict_latest('ETH', eth_ltf, df_htf=eth_htf)
                    eth_price = eth_ltf['close'].iloc[-1]
                    eth_latest_features = get_latest_ml_features('ETH', eth_ltf)
                    
                    if ml_proba_eth is not None:
                        proba_str = f"{ml_proba_eth:.3f}"
                    else:
                        proba_str = "None"
                    print(f"{now_str()} [ETH] Price: {eth_price:.2f}, ML: {proba_str}, ETH Trades: {trade_count_eth}/{eth_config['daily_cap']}")
                    sync_position_state('ETH', live_positions_map, eth_price)
                    
                    # === STEP 0: Calculate SMC Score early for logging ===
                    smc_score_early = 0
                    smc_details_early = {}
                    if eth_htf is not None and eth_ltf is not None:
                        smc_score_early, smc_details_early = calculate_smc_score_eth(eth_htf, eth_ltf)
                    
                    # === STEP 1: Hard Gates Check ===
                    gates_passed, gate_reason = check_eth_hard_gates(eth_ltf, eth_config)
                    
                    # Always show gates check
                    gates_log = f"[ETH GATES CHECK] {now_str()}\n"
                    gates_log += f"  Position Open: {'✗ YES (blocked)' if positions['ETH'] is not None else '✓ NO'}\n"
                    gates_log += f"  Daily Cap ETH: {trade_count_eth}/{eth_config['daily_cap']} {'✓' if trade_count_eth < eth_config['daily_cap'] else '✗'}\n"
                    gates_log += f"  Daily Cap Total: {trade_count}/{max_trades_total} {'✓' if trade_count < max_trades_total else '✗'}\n"
                    if eth_config['last_trade_time'] is not None:
                        time_since_last = (datetime.datetime.utcnow() - eth_config['last_trade_time']).total_seconds() / 60
                        gates_log += f"  Cooldown: {time_since_last:.1f}/{eth_config['cooldown_minutes']} min {'✓' if time_since_last >= eth_config['cooldown_minutes'] else '✗'}\n"
                    
                    if eth_ltf is not None and len(eth_ltf) >= 14:
                        atr = calculate_atr(eth_ltf['high'], eth_ltf['low'], eth_ltf['close'], period=14)
                        atr_ratio = atr.iloc[-1] / eth_price if eth_price > 0 else 0
                        gates_log += f"  ATR: {atr_ratio*100:.3f}% (min: 0.15%) {'✓' if atr_ratio >= 0.0015 else '✗'}\n"
                    
                    spread, bid, ask = calculate_spread(eth_config['symbol'])
                    if spread is not None and eth_price > 0:
                        spread_ratio = spread / eth_price
                        gates_log += f"  Spread: {spread_ratio*100:.3f}% (max: 0.05%) {'✓' if spread_ratio <= 0.0005 else '✗'}\n"
                    
                    gates_log += f"  Result: {'✓ PASSED' if gates_passed else f'✗ BLOCKED: {gate_reason}'}\n"
                    
                    # Add SMC points to gates log
                    smc_passed_count = sum(1 for k in ['htf_bias','bos','sweep','ob_proximity','discount_zone','fvg_touched'] if smc_details_early.get(k))
                    gates_log += f"  SMC points: {smc_passed_count}/6 {'✓' if smc_passed_count >= 4 else '✗'}"
                    print(gates_log)
                    
                    if gates_passed:
                        # === STEP 2: Get SMC Score ===
                        should_enter, smc_score, smc_details = get_buy_signal_eth_smc(eth_htf, eth_ltf)
                        
                        # Detailed ETH Strategy Logging
                        eth_log = f"\n[ETH ANALYSIS] {now_str()}\n"
                        eth_log += f"  Price: {eth_price:.2f}\n"
                        eth_log += f"  SMC Score: {smc_score}/6 (min required: 4) {'✓' if smc_score >= 4 else '✗'}\n"
                        passed_components = sum(1 for k in ['htf_bias','bos','sweep','ob_proximity','discount_zone','fvg_touched'] if smc_details.get(k))
                        eth_log += f"  SMC Components Passed: {passed_components}/6\n"
                        eth_log += f"    ├─ HTF Bias: {'✓' if smc_details.get('htf_bias') else '✗'}\n"
                        eth_log += f"    ├─ BOS: {'✓' if smc_details.get('bos') else '✗'}\n"
                        eth_log += f"    ├─ Sweep: {'✓' if smc_details.get('sweep') else '✗'}\n"
                        eth_log += f"    ├─ OB Proximity: {'✓' if smc_details.get('ob_proximity') else '✗'}\n"
                        eth_log += f"    ├─ Discount Zone: {'✓' if smc_details.get('discount_zone') else '✗'}\n"
                        eth_log += f"    └─ FVG Touched: {'✓' if smc_details.get('fvg_touched') else '✗'}\n"
                        
                        # ATR Check
                        if eth_ltf is not None and len(eth_ltf) >= 14:
                            atr = calculate_atr(eth_ltf['high'], eth_ltf['low'], eth_ltf['close'], period=14)
                            atr_ratio = atr.iloc[-1] / eth_price if eth_price > 0 else 0
                            eth_log += f"  ATR: {atr_ratio*100:.3f}% (min: 0.15%) {'✓' if atr_ratio >= 0.0015 else '✗'}\n"
                        
                        # Spread Check
                        spread, bid, ask = calculate_spread(eth_config['symbol'])
                        if spread is not None and eth_price > 0:
                            spread_ratio = spread / eth_price
                            eth_log += f"  Spread: {spread_ratio*100:.3f}% (max: 0.05%) {'✓' if spread_ratio <= 0.0005 else '✗'}\n"
                        
                        # News Sentiment
                        news_sentiment = get_news_sentiment()
                        eth_log += f"  News Sentiment: {news_sentiment if news_sentiment is not None else 'None'} {'✓' if news_sentiment is None or news_sentiment > -60 else '⚠️ BAD'}\n"
                        
                        # ML Score
                        eth_log += f"  ML Score: {proba_str}\n"
                        eth_log += f"  Trades: {trade_count}/{max_trades_total}, ETH: {trade_count_eth}/{eth_config['daily_cap']}\n"
                        
                        print(eth_log)
                        
                        if should_enter and smc_score >= 4:
                            # === STEP 3: ML Gating (Smart Threshold) ===
                            ml_passed = False
                            ml_threshold_used = 0.0
                            
                            # Check news sentiment first
                            news_allows_entry = True
                            
                            if news_sentiment is not None and news_sentiment <= -60:
                                # Bad news - only allow if SMC >= 5 AND ML >= 0.62
                                if smc_score >= 5 and ml_proba_eth and ml_proba_eth >= 0.62:
                                    ml_passed = True
                                    ml_threshold_used = 0.62
                                    news_allows_entry = True
                                    print(f"[ETH] ⚠️  Bad news ({news_sentiment}) overridden by strong SMC (score={smc_score}, ML={ml_proba_eth:.3f})")
                                else:
                                    news_allows_entry = False
                                    print(f"[ETH] ❌ Bad news blocks entry (sentiment={news_sentiment}, need SMC>=5 & ML>=0.62)")
                            else:
                                # Normal conditions - use adaptive threshold
                                if smc_score >= 5:
                                    # Strong SMC - lower threshold
                                    ml_threshold_used = eth_config['ml_threshold_strong']  # 0.52
                                    if ml_proba_eth and ml_proba_eth >= ml_threshold_used:
                                        ml_passed = True
                                elif smc_score == 4:
                                    # Medium SMC - base threshold
                                    ml_threshold_used = eth_config['ml_threshold_base']  # 0.58
                                    if ml_proba_eth and ml_proba_eth >= ml_threshold_used:
                                        ml_passed = True

                            
                            # === STEP 4: Final Entry Decision ===
                            ml_decision_log = f"[ETH ML DECISION]\n"
                            ml_decision_log += f"  SMC Score: {smc_score} (required: >=4)\n"
                            ml_decision_log += f"  ML Score: {ml_proba_eth:.3f}\n"
                            ml_decision_log += f"  ML Threshold: {ml_threshold_used:.2f}\n"
                            ml_decision_log += f"  News Sentiment: {news_sentiment}\n"
                            ml_decision_log += f"  ML Passed: {'✓ YES' if ml_passed else '✗ NO'}\n"
                            ml_decision_log += f"  News OK: {'✓ YES' if news_allows_entry else '✗ NO'}\n"
                            ml_decision_log += f"  US Market: {'✓ CLOSED' if not is_us_market_open_window() else '✗ OPEN'}"
                            print(ml_decision_log)
                            
                            if ml_passed and news_allows_entry and not is_us_market_open_window():

                                # Final live check
                                if positions['ETH'] is not None:
                                    print(f"[ETH] ⚠️  Position already exists (local), skipping buy")
                                    continue
                                
                                if has_open_position_live(eth_config['symbol']):
                                    print(f"[ETH] ⚠️  LIVE CHECK: Position exists on exchange, skipping buy")
                                    continue
                                
                                # Log SMC details
                                smc_log = f"[ETH] 🎯 ENTRY SIGNAL!\n"
                                smc_log += f"  SMC Score: {smc_score}/6\n"
                                smc_log += f"  HTF Bias: {'✓' if smc_details.get('htf_bias') else '✗'}\n"
                                smc_log += f"  BOS: {'✓' if smc_details.get('bos') else '✗'}\n"
                                smc_log += f"  Sweep: {'✓' if smc_details.get('sweep') else '✗'}\n"
                                smc_log += f"  OB Proximity: {'✓' if smc_details.get('ob_proximity') else '✗'}\n"
                                smc_log += f"  Discount: {'✓' if smc_details.get('discount_zone') else '✗'}\n"
                                smc_log += f"  FVG: {'✓' if smc_details.get('fvg_touched') else '✗'}\n"
                                smc_log += f"  ML: {ml_proba_eth:.3f} (threshold: {ml_threshold_used:.2f})\n"
                                smc_log += f"  Trades: {trade_count}/{max_trades_total}, ETH: {trade_count_eth}/{eth_config['daily_cap']}"
                                print(smc_log)
                                
                                try:
                                    # Mark position as pending
                                    positions['ETH'] = 'PENDING'
                                    
                                    buy_order = place_order(eth_config['symbol'], 'buy', eth_config['amount'])
                                    if buy_order is None:
                                        print("[ETH] Buy order failed, skipping")
                                        positions['ETH'] = None
                                    else:
                                        entry_price = buy_order.get('average')
                                        if entry_price is None:
                                            entry_price = eth_price
                                        
                                        sl_id = place_stop_loss(eth_config['symbol'], entry_price, eth_config['amount'], eth_config['sl_percent'])
                                        tp_ids = place_take_profit_orders(eth_config['symbol'], entry_price, eth_config['amount'], tp_tiers)
                                        
                                        positions['ETH'] = {
                                            'side': 'long',
                                            'amount': eth_config['amount'],
                                            'entry_price': entry_price,
                                            'entry_time': datetime.datetime.utcnow().isoformat(),
                                            'tp_ids': tp_ids,
                                            'sl_id': sl_id,
                                            'smc_score': smc_score,
                                            'ml_features': eth_latest_features
                                        }
                                        trade_count += 1
                                        trade_count_eth += 1
                                        eth_config['last_trade_time'] = datetime.datetime.utcnow()
                                        
                                        print(f"[ETH] ✅ Position opened: entry={entry_price:.2f}, count={trade_count}, eth_count={trade_count_eth}")
                                        
                                        msg = f"🚀 ETH BUY (SMC {smc_score}/6): {eth_config['amount']} @ ${entry_price:.2f}, ML:{ml_proba_eth:.2f}, Trade #{trade_count}/14"
                                        if bot:
                                            send_telegram_message(msg)
                                        
                                        _append_log('buy', 'ETH', {
                                            'price': entry_price, 
                                            'amount': eth_config['amount'],
                                            'smc_score': smc_score,
                                            'ml_proba': float(ml_proba_eth) if ml_proba_eth else None,
                                            'smc_details': smc_details
                                        })
                                except Exception as e:
                                    print(f"[ERROR] ETH buy logic: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    _append_log('error', 'ETH', {'error': str(e)})
                            else:
                                # Log why entry was blocked
                                if not ml_passed:
                                    print(f"[ETH] ML Gate: {ml_proba_eth:.3f} < {ml_threshold_used:.2f} (SMC score: {smc_score})")
                                if not news_allows_entry:
                                    print(f"[ETH] News blocks entry")
                        else:
                            # SMC score too low or no entry trigger
                            if not should_enter:
                                pass  # Entry trigger not met
                            elif smc_score < 4:
                                print(f"[ETH] SMC score too low: {smc_score}/6")
                    else:
                        # Gates failed
                        if gate_reason and "Cooldown" not in gate_reason and "Position" not in gate_reason:
                            print(f"[ETH] Gate failed: {gate_reason}")
                    
                except Exception as e:
                    print(f"[ERROR] ETH strategy: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"{now_str()} [ETH] Data fetch failed (HTF or LTF)")
                sync_position_state('ETH', live_positions_map, None)
            
            # Note: ETH sell signals removed - using TP/SL orders only
            
            time.sleep(60)
        
        except Exception as e:
            print(f"[ERROR] {e}")
            _append_log('error', 'SYSTEM', {'error': str(e)})
            time.sleep(60)

if __name__ == '__main__':
    main_loop()

