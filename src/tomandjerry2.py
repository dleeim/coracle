import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

# 1) Fetch hourly data from 2022-01-01 to 2025-05-02 with paging
exchange = ccxt.bybit({'options': {'defaultType': 'future'}})
symbol   = 'BTC/USDT:USDT'

start_iso = '2024-01-01T00:00:00Z'
end_iso   = '2025-05-02T00:00:00Z'
since     = exchange.parse8601(start_iso)
end_ts    = exchange.parse8601(end_iso)
all_bars  = []

while since < end_ts:
    bars = exchange.fetch_ohlcv(symbol, timeframe='1h', since=since, limit=2000)
    if not bars:
        break
    all_bars += bars
    # advance 'since' to timestamp of last fetched bar + 1ms
    last_ts = bars[-1][0]
    since   = last_ts + 1

# build dataframe
df = pd.DataFrame(all_bars, columns=['timestamp','open','high','low','close','volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
df.set_index('timestamp', inplace=True)

# trim any overshoot past end_iso
df = df.loc[:end_iso]

# 2) Daily close returns (UTC 00:00)
daily_close = df['close'].resample('24h').last()
daily_ret   = daily_close.pct_change()

# 3) Build 12h long/short signals and trading operation

# parameters
threshold = 0.002
stop_loss  = 0.02    # 1% stop
take_profit= 0.05    # 2% target
leverage   = 5

# new container for adjusted signals (will be +5, -5, or 0)
signals = pd.Series(0, index=df.index)

# we'll also track forced exit times to avoid double‐counting
forced_exit = set()

for ts, r in daily_ret.dropna().items():
    start = ts + pd.Timedelta(hours=1)
    if start not in df.index:
        continue

    # decide direction
    if   r >  threshold: direction =  1
    elif r < -threshold: direction = -1
    else: continue

    entry_price = df.at[start, 'open']  # or 'close', your choice
    sl_price    = entry_price * (1 - direction*stop_loss)
    tp_price    = entry_price * (1 + direction*take_profit)

    # walk forward up to 12 hours
    for t in df.loc[start : start+pd.Timedelta(hours=12)].index:
        # skip if we've already force‐exited
        if t in forced_exit:
            break

        hi = df.at[t, 'high']
        lo = df.at[t, 'low']

        # check TP/SL
        if direction ==  1 and hi  >= tp_price:
            exit_time = t
            forced_exit.add(t)
            break
        if direction ==  1 and lo   <= sl_price:
            exit_time = t
            forced_exit.add(t)
            break
        if direction == -1 and lo   <= tp_price:
            exit_time = t
            forced_exit.add(t)
            break
        if direction == -1 and hi   >= sl_price:
            exit_time = t
            forced_exit.add(t)
            break
    else:
        # no SL/TP hit → exit at 12h close
        exit_time = start + pd.Timedelta(hours=12)

    # now assign your signal between entry and exit
    signals.loc[start:exit_time] = direction * leverage

# 4) Hourly strategy returns before fees
hr_ret    = df['close'].pct_change().fillna(0)
strat_ret = signals * hr_ret

# 5) Define VIP-0 fees and apply at each entry/exit
maker_fee = 0.00020   # 0.0200%
taker_fee = 0.00055   # 0.0550%
# entry = when signal goes from 0 → ±1; exit = when signal goes ±1 → 0
entry_idx = signals[(signals != 0) & (signals.shift(1) == 0)].index
exit_idx  = signals[(signals != 0) & (signals.shift(-1).fillna(0) == 0)].index
for t in entry_idx:
    strat_ret.loc[t] -= taker_fee
for t in exit_idx:
    strat_ret.loc[t] -= maker_fee

# 6) Net equity curve
equity = (1 + strat_ret).cumprod()

# 7) Recompute per‐trade net returns
group_id   = (signals != signals.shift()).cumsum()
trade_net  = []
fee_rt     = maker_fee + taker_fee
for _, grp in signals[signals != 0].groupby(group_id):
    pos   = grp.iloc[0]
    gross = (1 + hr_ret.loc[grp.index] * pos).prod() - 1
    net   = (1 + gross) * (1 - fee_rt) - 1
    trade_net.append(net)
trade_net = pd.Series(trade_net)

# categorize wins/losses
wins      = trade_net[trade_net > 0]
losses    = trade_net[trade_net <= 0]
n_trades  = len(trade_net)

# 8) Performance metrics
total_ret     = equity.iloc[-1] - 1
days          = (equity.index[-1] - equity.index[0]).days
ann_ret       = equity.iloc[-1] ** (365.0/days) - 1
ann_vol       = strat_ret.std() * np.sqrt(24*365)
sharpe        = ann_ret / ann_vol if ann_vol else np.nan
mdd           = ((equity - equity.cummax())/equity.cummax()).min()

win_prob      = len(wins)  / n_trades
avg_win       = wins.mean()
lose_prob     = len(losses) / n_trades
avg_loss      = losses.mean()
profit_factor = wins.sum() / (-losses.sum()) if losses.sum() != 0 else np.nan
expectancy    = trade_net.mean()
calmar        = ann_ret / abs(mdd) if mdd < 0 else np.nan

# 9) Absolute P&L on 1_000 USDT
initial_cap = 1_000
final_cap   = initial_cap * equity.iloc[-1]
abs_profit  = final_cap - initial_cap

# 10) Print results
print(f"Backtest window:       {start_iso} → {end_iso}")
print(f"Trades (#):            {n_trades}")
print(f"Win %:                 {win_prob:.2%}")
print(f"Avg win per trade:     {avg_win:.2%}")
print(f"Lose %:                {lose_prob:.2%}")
print(f"Avg loss per trade:    {avg_loss:.2%}")
print(f"Expectancy (per trade):{expectancy:.2%}")
print(f"Profit factor:         {profit_factor:.2f}")
print(f"Calmar ratio:          {calmar:.2f}\n")

print(f"Total return:          {total_ret:.2%}")
print(f"Ann. return:           {ann_ret:.2%}")
print(f"Ann. vol (est):        {ann_vol:.2%}")
print(f"Sharpe ratio:          {sharpe:.2f}")
print(f"Max drawdown:          {mdd:.2%}\n")

print(f"Initial capital:       {initial_cap:.2f} USDT")
print(f"Final capital:         {final_cap:.2f} USDT")
print(f"Absolute profit:       {abs_profit:.2f} USDT")

# 11) Plot equity
plt.figure(figsize=(10,5))
plt.plot(equity.index, equity.values, label='Net-Fee Equity')
plt.title('12h Lead-Lag Momentum (2022-01-01 to 2025-05-02)')
plt.xlabel('Date'); plt.ylabel('Capital Multiple')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.show()