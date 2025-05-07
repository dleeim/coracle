#!/usr/bin/env python3
import os
import urllib.parse
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb

# ─── 1) LOAD MODEL ────────────────────────────────────────────────────────────────
pkg   = joblib.load('xgb_multi_tf_module.pkl')
model = pkg['model']
bst   = model.get_booster()
feature_names = bst.feature_names  # exact list & order

# ─── 2) SETUP DB & LOADER ───────────────────────────────────────────────────────
from xgb_multi_tf_pipeline import MultiTFLoader

load_dotenv()
user   = os.getenv("MYSQL_USER")
pw     = urllib.parse.quote_plus(os.getenv("MYSQL_PASSWORD"))
host   = os.getenv("MYSQL_HOST")
port   = os.getenv("MYSQL_PORT")
db     = os.getenv("MYSQL_DATABASE")
engine = create_engine(f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}")

base_tf = '1h'
tfs     = ['5m','15m','30m','1h','4h','6h','12h','1d']
ta_inds = [
    'SAR_real','ATR_real','RSI_real',
    'BBANDS_','EMA_real','MACD_','OBV_real'
]

loader = MultiTFLoader(engine, base_tf, tfs, ta_inds)

# ─── 3) BUILD FEATURE MATRIX ────────────────────────────────────────────────────
df = loader.transform(None)

# ─── 4) CLEAN & ALIGN FEATURES ──────────────────────────────────────────────────
# drop any rows missing the features the booster expects (or the price)
df = df.dropna(subset=feature_names + ['close'])

# reorder to booster.feature_names exactly
X = df.loc[:, feature_names]

# ─── 5) PREDICTION & SIGNAL ─────────────────────────────────────────────────────
dtest     = xgb.DMatrix(X)
df['proba'] = bst.predict(dtest)
# 0.5 cutoff
df['signal'] = (df['proba'] > 0.5).astype(int)

# ─── 6) RAW RETURNS ──────────────────────────────────────────────────────────────
df['ret']       = df['close'].pct_change().shift(-1)
df['strat_ret'] = df['signal'] * df['ret']
# fill any NaNs so equity curve is defined throughout
strat_ret = df['strat_ret'].fillna(0)

# ─── 7) APPLY FEES ───────────────────────────────────────────────────────────────
maker_fee = 0.00020   # 0.0200%
taker_fee = 0.00055   # 0.0550%

signals   = df['signal']
fee_ret   = strat_ret.copy()

# entry = 0→1, exit = 1→0
entry_idx = signals[(signals==1)&(signals.shift(1)==0)].index
exit_idx  = signals[(signals==1)&(signals.shift(-1).fillna(0)==0)].index

for t in entry_idx: fee_ret.loc[t] -= taker_fee
for t in exit_idx:  fee_ret.loc[t] -= maker_fee

# ─── 8) EQUITY CURVE ────────────────────────────────────────────────────────────
equity = (1 + fee_ret).cumprod()

# ─── 9) PER-TRADE NET RETURNS ───────────────────────────────────────────────────
group_id  = (signals != signals.shift()).cumsum()
trade_net = []
total_fee = maker_fee + taker_fee

for _, grp in signals[signals==1].groupby(group_id):
    pos     = grp.iloc[0]  # always 1 here
    gross   = (1 + df['ret'].loc[grp.index] * pos).prod() - 1
    net     = (1 + gross) * (1 - total_fee) - 1
    trade_net.append(net)

trade_net = pd.Series(trade_net, name='net_return')

# ─── 10) PERFORMANCE METRICS ────────────────────────────────────────────────────
wins      = trade_net[trade_net>0]
losses    = trade_net[trade_net<=0]
n_trades  = len(trade_net)

# time-based
start_date = equity.index[0]
end_date   = equity.index[-1]
days       = max((end_date - start_date).days, 1)

total_ret  = equity.iloc[-1] - 1
ann_ret    = equity.iloc[-1]**(365.0/days) - 1
ann_vol    = fee_ret.std() * np.sqrt(24*365)
sharpe     = ann_ret / ann_vol if ann_vol else np.nan
mdd        = ((equity - equity.cummax())/equity.cummax()).min()

# trade-based
win_rate      = len(wins) / n_trades if n_trades else np.nan
avg_win       = wins.mean() if len(wins) else np.nan
avg_loss      = losses.mean() if len(losses) else np.nan
profit_factor = wins.sum() / (-losses.sum()) if losses.sum()!=0 else np.nan
expectancy    = trade_net.mean() if n_trades else np.nan
calmar        = ann_ret / abs(mdd) if mdd<0 else np.nan

# absolute P&L on 200 USDT
initial_cap = 200.0
final_cap   = initial_cap * equity.iloc[-1]
abs_profit  = final_cap - initial_cap

# ─── 11) PRINT SUMMARY ──────────────────────────────────────────────────────────
print(f"Backtest: {start_date.date()} → {end_date.date()}")
print(f"Trades:           {n_trades}")
print(f"Win rate:         {win_rate:.2%}")
print(f"Avg win:          {avg_win:.2%}")
print(f"Avg loss:         {avg_loss:.2%}")
print(f"Expectancy:       {expectancy:.2%}")
print(f"Profit factor:    {profit_factor:.2f}")
print(f"Calmar ratio:     {calmar:.2f}\n")

print(f"Total return:     {total_ret:.2%}")
print(f"Annual return:    {ann_ret:.2%}")
print(f"Annual vol:       {ann_vol:.2%}")
print(f"Sharpe ratio:     {sharpe:.2f}")
print(f"Max drawdown:     {mdd:.2%}\n")

print(f"Initial capital:  {initial_cap:.2f} USDT")
print(f"Final capital:    {final_cap:.2f} USDT")
print(f"Absolute profit:  {abs_profit:.2f} USDT")

# ─── 12) PLOT EQUITY ─────────────────────────────────────────────────────────────
plt.figure(figsize=(10,5))
plt.plot(equity.index, equity, label='Net‐Fee Equity')
plt.title(f'Equity Curve {start_date.date()}→{end_date.date()}')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (×)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
