#!/usr/bin/env python
import os
import ccxt
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

api_key = os.getenv('API_KEY')
secret = os.getenv('SECRET')

if not api_key or not secret:
    print("❌ Error: API_KEY and SECRET not found")
    exit(1)

exchange = ccxt.krakenfutures({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
})

print("🔍 Checking Open Positions on Kraken Futures\n")
print("=" * 60)

try:
    # Method 1: Fetch positions
    positions = exchange.fetch_positions()
    
    print(f"\n📊 Total positions returned: {len(positions)}\n")
    
    open_positions = []
    for pos in positions:
        contracts = pos.get('contracts', 0)
        if contracts is not None and contracts != 0:
            open_positions.append(pos)
    
    if not open_positions:
        print("✅ No open positions found")
    else:
        print(f"🎯 Found {len(open_positions)} open position(s):\n")
        
        for i, pos in enumerate(open_positions, 1):
            symbol = pos.get('symbol', 'UNKNOWN')
            contracts = pos.get('contracts', 0)
            side = pos.get('side', 'UNKNOWN')
            entry_price = pos.get('entryPrice', 0)
            notional = pos.get('notional')
            unrealized_pnl = pos.get('unrealizedPnl')
            percentage = pos.get('percentage')
            leverage = pos.get('leverage', 0)
            
            print(f"Position #{i}:")
            print(f"  Symbol: {symbol}")
            print(f"  Side: {side}")
            print(f"  Contracts: {contracts}")
            print(f"  Entry Price: ${entry_price:,.2f}" if entry_price else "  Entry Price: N/A")
            if notional is not None:
                print(f"  Notional: ${notional:,.2f}")
            if unrealized_pnl is not None:
                pnl_str = f"  Unrealized PnL: ${unrealized_pnl:,.2f}"
                if percentage is not None:
                    pnl_str += f" ({percentage:+.2f}%)"
                print(pnl_str)
            print(f"  Leverage: {leverage}x" if leverage else "  Leverage: N/A")
            print()
    
    # Method 2: Check recent orders (last 1 hour)
    print("\n" + "=" * 60)
    print("📋 Checking Recent Orders (Last Hour)\n")
    
    symbols = ['BTC/USD:USD', 'ETH/USD:USD']
    for symbol in symbols:
        try:
            orders = exchange.fetch_orders(symbol, limit=10)
            recent_buys = []
            now = datetime.now(timezone.utc).timestamp() * 1000
            one_hour_ago = now - (60 * 60 * 1000)
            
            for order in orders:
                if order['timestamp'] >= one_hour_ago and order['side'] == 'buy' and order['status'] == 'closed':
                    recent_buys.append(order)
            
            if recent_buys:
                print(f"{symbol}:")
                for order in recent_buys:
                    order_time = datetime.fromtimestamp(order['timestamp'] / 1000, tz=timezone.utc)
                    print(f"  ✅ BUY: {order['amount']} @ ${order.get('average', order.get('price', 0)):,.2f}")
                    print(f"     Time: {order_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print()
        except Exception as e:
            print(f"{symbol}: Could not fetch orders - {e}\n")
    
    print("=" * 60)
    print("\n✅ Position check complete")

except Exception as e:
    print(f"\n❌ Error checking positions: {e}")
    import traceback
    traceback.print_exc()
