# Kraken Futures Trading Bot

Dual-asset (BTC+ETH) automated trading bot for Kraken Futures with ML-based entry gating, news sentiment filtering, and 3-tier take-profit system.

## Features

- **Dual Asset Trading**: BTC (0.0018) + ETH (0.045) with shared trade limit (max 6 concurrent)
- **Technical Indicators**: RSI, MACD, Stochastic, ATR-based entry/exit signals
- **ML Gating**: Logistic regression model (threshold 0.55) trained every 30min
- **News Filter**: CryptoPanic sentiment analysis with 30min cooldown on negative news
- **Risk Management**: 
  - BTC: 1.5% stop loss
  - ETH: 1.0% stop loss
  - 3-tier take profit: 0.4% (40%), 0.8% (30%), 1.5% (30%)
- **Market Timing**: Avoids US market open window (14:30 UTC ±15min)
- **Telegram Notifications**: Real-time trade alerts and status updates

## Latest Updates (Jan 24, 2026)

### 🎯 MAJOR UPDATE: True SMC Strategy for ETH

**Complete rewrite of ETH strategy from pseudo-SMC to institutional-grade Smart Money Concepts:**

#### New SMC Components Implemented:
1. **Market Structure Analysis**
   - Break of Structure (BOS) detection for trend confirmation
   - Change of Character (CHOCH) identification

2. **Order Block Recognition** 
   - Detects institutional entry zones (high volume + impulsive moves)
   - Tracks bullish and bearish order blocks separately
   - Validates price proximity to OB before entry

3. **Liquidity Sweep Detection**
   - Identifies when price breaks previous highs/lows to trigger institutional stops
   - Confirmation of smart money entry setups

4. **Fair Value Gap (FVG) Identification**
   - Detects imbalances in price action
   - Uses gaps as take-profit targets (price naturally fills FVGs)
   - Bullish and bearish FVG tracking

5. **Premium vs Discount Zone Analysis**
   - Classifies current price location (upper 30%, lower 30%, neutral)
   - Buys only in discount zones (higher probability entries)
   - Sells when reaching premium zones with confirmation

6. **Breaker Block Concept**
   - Failed order blocks become resistance/support
   - Reversal confirmation at breaker block levels

7. **Multi-Timeframe Confirmation**
   - 4-hour bias analysis: uses HTF for macro direction
   - Only takes LTF entries when HTF bias is bullish
   - Top-down approach like institutional traders

#### ETH Entry Criteria (ALL must be satisfied):
- ✓ Bullish BOS on 1h timeframe
- ✓ Liquidity sweep low detected (smart money confirmation)
- ✓ Order block support identified and price near it
- ✓ In discount zone (lower 30% of recent range)
- ✓ Fair Value Gap present (for target levels)
- ✓ Volume confirmation (>80% of average)
- ✓ ML model probability ≥ 0.55
- ✓ 4-hour timeframe showing bullish bias
- ✓ News sentiment not blocking

#### ETH Exit Criteria (ANY of these):
- Price reaches FVG level (takes profit at imbalance)
- Bearish BOS detected (structure reversal)
- Price in premium zone with volume confirmation
- Breaker block rejection with reversal candle
- RSI > 85 (extreme overbought)

**Note**: BTC strategy remains unchanged (indicator-based RSI/MACD/Stoch)

### Previous Bug Fixes (Jan 23)
- **Duplicate Entry Prevention**: Added double-check and PENDING state to prevent multiple position entries
- **Stop Loss Improvements**: 
  - Fixed price precision (ETH: 1 decimal, BTC: 0 decimals)
  - Added `reduceOnly: True` flag to prevent SL order errors
  - Enhanced error logging for invalidPrice rejections
- **Position State Management**: More robust state tracking with explicit confirmation messages

## Setup

1. Create virtual environment: `python -m venv venv`
2. Activate: `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Configure environment variables:
   - `KRAKEN_API_KEY`: Your Kraken Futures API key
   - `KRAKEN_API_SECRET`: Your Kraken Futures API secret
   - `TELEGRAM_BOT_TOKEN`: Bot token from @BotFather
   - `TELEGRAM_CHAT_ID`: Your chat ID (get from @userinfobot)
   - `CRYPTOPANIC_API_KEY`: CryptoPanic API token

## Usage

Run the bot:
```bash
python src/main.py > bot.log 2>&1 &
tail -f bot.log
```

Monitor logs:
```bash
# View recent trades
tail -100 bot.log | grep -E "(BUY|SELL|SL|TP)"

# Check ML predictions
tail -100 bot.log | grep "ML:"

# View position status
tail -50 bot.log | grep "Position"
```

## Configuration

Edit `src/main.py` to adjust:
- `max_trades_total`: Maximum concurrent trades (default: 6)
- `ml_threshold`: ML probability threshold for entry (default: 0.55)
- `sl_percent`: Stop loss percentages (BTC: 1.5%, ETH: 1.0%)
- `tp_tiers`: Take profit tiers and percentages

## Requirements

- Python 3.9+
- Kraken Futures account with API keys
- Telegram bot (optional but recommended)
- CryptoPanic API key (optional)

## Important Notes

- Bot only opens LONG positions (no SHORT trades)
- All timestamps displayed in Israel time (UTC+2)
- Stop loss orders use `reduceOnly` flag to prevent position reversal
- ML models auto-train every 30 minutes with latest 85 samples
- News cooldown activates for 30 minutes on negative sentiment detection