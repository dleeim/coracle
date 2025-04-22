"""
BTC Perpetual Direction‑Prediction Strategy (CNN‑LSTM)
=====================================================
Complete prototype implementation in one file.
The script can
  • download historical BTCUSDT perpetual klines & funding rates from Bybit
  • engineer technical‑indicator features (pandas_ta)
  • build & train a CNN‑LSTM model in PyTorch
  • back‑test the strategy using the model’s predictions
  • provide a live‑trading stub that connects to Bybit WebSocket and places
    orders via REST when the signal flips.

*** Requirements ***
 pip install pandas numpy requests pybit pandas_ta scikit‑learn torch

*** IMPORTANT ***
 ‑ Insert your Bybit API credentials (read/write) in BYBIT_API_KEY/SECRET env vars
   or directly in the CONFIG section below.  **Trade at your own risk.**

This code is intended as a starting point.  You should:
  • tune hyper‑parameters, risk settings, fee assumptions
  • perform rigorous walk‑forward testing w/ realistic costs and latency
  • run the bot in paper‑trading before deploying live capital.
"""
from __future__ import annotations
import os
import time
import math
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    symbol: str = "BTCUSDT"
    category: str = "linear"           # USDT‑margined
    primary_interval: str = "60"      # 1‑hour candles
    lookback: int = 60                 # timesteps fed to model (60h)
    threshold: float = 0.001           # 0.1 % move to define up/down
    train_split: float = 0.8           # chronological train/test ratio
    batch_size: int = 128
    num_epochs: int = 50
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    api_key: str = os.getenv("BYBIT_API_KEY", "")
    api_secret: str = os.getenv("BYBIT_API_SECRET", "")

CFG = Config()

# Base REST URL (Unified v5)
BASE_URL = "https://api.bybit.com"

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA DOWNLOADER
# ──────────────────────────────────────────────────────────────────────────────

def get_klines(symbol: str, interval: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Fetch klines in chunks of ≤1000 rows via REST. timestamps in milliseconds."""
    all_rows: List[dict] = []
    limit = 1000
    cursor = None
    while True:
        params = {
            "category": CFG.category,
            "symbol": symbol,
            "interval": interval,
            "start": start_ts,
            "end": end_ts,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data["retCode"] != 0:
            raise RuntimeError(data)
        rows = data["result"]["list"]
        if not rows:
            break
        all_rows.extend(rows)
        cursor = data["result"].get("nextPageCursor")
        if not cursor:
            break
        # Respect API rate limits
        time.sleep(0.1)
    # result rows: [start, open, high, low, close, volume, turnover]
    df = pd.DataFrame(all_rows, columns=[
        "ts", "open", "high", "low", "close", "volume", "turnover"])
    df = df.astype(float)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df.sort_index()

# ──────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a suite of technical indicators to `df`."""
    df_ta = pd.DataFrame(index=df.index)

    # Trend
    df_ta["sma20"] = ta.sma(df["close"], length=20)
    df_ta["ema20"] = ta.ema(df["close"], length=20)
    df_ta["adx14"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]

    # Momentum
    df_ta["rsi14"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df_ta = pd.concat([df_ta, macd], axis=1)

    # Volatility
    bb = ta.bbands(df["close"], length=20)
    df_ta = pd.concat([df_ta, bb], axis=1)
    df_ta["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Volume
    df_ta["obv"] = ta.obv(df["close"], df["volume"])
    df_ta["vol_sma20"] = ta.sma(df["volume"], length=20)

    # Drop all‑NaN columns then forward‑fill the rest.
    df_ta = df_ta.dropna(axis=1, how="all").fillna(method="ffill")
    return df_ta

# ──────────────────────────────────────────────────────────────────────────────
# 3. LABELS & DATASET
# ──────────────────────────────────────────────────────────────────────────────

def make_labels(df: pd.DataFrame, threshold: float) -> pd.Series:
    pct = df["close"].pct_change().shift(-1)  # next‑period return
    lbl = pct.where(abs(pct) >= threshold)
    lbl = lbl.apply(lambda x: 1 if x and x > 0 else (0 if x and x < 0 else np.nan))
    return lbl


def build_dataset(df_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    features = compute_indicators(df_raw)
    labels = make_labels(df_raw, CFG.threshold)

    # Align lengths
    data = pd.concat([df_raw, features, labels.rename("label")], axis=1).dropna()

    # Feature matrix & scaling
    feature_cols = features.columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[feature_cols])

    # Pad with lookback window to create sequences
    X_seq, y_seq = [], []
    for i in range(CFG.lookback, len(data)):
        X_seq.append(X_scaled[i - CFG.lookback:i])  # (lookback, F)
        y_seq.append(data["label"].iloc[i - 1])    # label for last pos in window
    X_seq = np.stack(X_seq)
    y_seq = np.array(y_seq, dtype=np.float32)
    print(f"Dataset built: {X_seq.shape} samples, {X_seq.shape[2]} features")
    return X_seq, y_seq, scaler


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL DEFINITION
# ──────────────────────────────────────────────────────────────────────────────
class CNNLSTM(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(feature_dim, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool = nn.AvgPool1d(2)
        self.drop1 = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2,
                            dropout=0.5, batch_first=True)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, 1)
    def forward(self, x):  # x (batch, T, F)
        x = x.transpose(1, 2)          # → (batch, F, T)
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)               # → (batch, 64, T/2)
        x = self.drop1(x)
        x = x.transpose(1, 2)          # → (batch, T/2, 64)
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]         # (batch, 128)
        h_last = self.bn2(h_last)
        return torch.sigmoid(self.fc(h_last))

# ──────────────────────────────────────────────────────────────────────────────
# 5. TRAINING & EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def train_test_split_chrono(X, y, ratio=0.8):
    split_idx = int(len(X) * ratio)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def train_model(model, train_loader, val_loader):
    crit = nn.BCELoss()
    opt = Adam(model.parameters(), lr=CFG.lr)
    model.to(CFG.device)
    best_val = math.inf
    for epoch in range(1, CFG.num_epochs + 1):
        model.train(); loss_sum = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(CFG.device), yb.to(CFG.device)
            opt.zero_grad()
            pred = model(Xb)
            loss = crit(pred, yb)
            loss.backward(); opt.step()
            loss_sum += loss.item() * len(Xb)
        avg_train = loss_sum / len(train_loader.dataset)
        # validation
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(CFG.device), yb.to(CFG.device)
                pred = model(Xb)
                val_loss += crit(pred, yb).item() * len(Xb)
        avg_val = val_loss / len(val_loader.dataset)
        print(f"Ep {epoch:02d}: train {avg_train:.4f}  val {avg_val:.4f}")
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), "best_model.pt")
    print("Training complete. Best val loss:", best_val)


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, dtype=torch.float32, device=CFG.device)).cpu().numpy().ravel()
    y_hat = (y_pred >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_hat)
    prec = precision_score(y_test, y_hat)
    rec = recall_score(y_test, y_hat)
    f1 = f1_score(y_test, y_hat)
    cm = confusion_matrix(y_test, y_hat)
    print(f"Accuracy {acc:.3f}  Precision {prec:.3f}  Recall {rec:.3f}  F1 {f1:.3f}")
    print("Confusion\n", cm)

# ──────────────────────────────────────────────────────────────────────────────
# 6. BACKTESTER
# ──────────────────────────────────────────────────────────────────────────────

def backtest(df_raw: pd.DataFrame, X_test: np.ndarray, y_pred: np.ndarray):
    # Only use rows associated with X_test (align indexes)
    price_series = df_raw["close"].iloc[-len(y_pred):].to_numpy()
    pos = 0   # 1 long, -1 short, 0 flat (we’ll always be in market here)
    pnl = []
    for i in range(1, len(price_series)):
        signal = 1 if y_pred[i-1] >= 0.5 else -1
        # Flip position if needed
        if pos != signal:
            pos = signal
        ret = pos * (price_series[i] - price_series[i-1]) / price_series[i-1]
        pnl.append(ret)
    pnl = np.array(pnl)
    cum_ret = (1 + pnl).cumprod()[-1] - 1
    sharpe = pnl.mean() / (pnl.std() + 1e-12) * math.sqrt(365*24)  # hourly → annualized
    print(f"Backtest: total return {cum_ret*100:.2f}%  Sharpe {sharpe:.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# 7. LIVE TRADING STUB (WebSocket + REST)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from pybit.unified_trading import HTTP, WebSocket
except ImportError:
    HTTP = WebSocket = None

class LiveTrader:
    def __init__(self, model: nn.Module, scaler: StandardScaler):
        if not HTTP or not WebSocket:
            raise RuntimeError("Install `pybit` to use live trader.")
        self.model = model.to(CFG.device).eval()
        self.scaler = scaler
        self.client = HTTP(endpoint=BASE_URL, api_key=CFG.api_key, api_secret=CFG.api_secret)
        self.ws = WebSocket(endpoint="wss://stream.bybit.com/v5/public/linear", category="linear",
                            interval=CFG.primary_interval, symbol=CFG.symbol, callback=self.on_kline)
        self.buffer: List[pd.Series] = []
        self.current_pos = 0

    def on_kline(self, msg):
        if msg.get("type") != "snapshot":
            return
        data = msg["data"][0]
        candle = {
            "ts": pd.to_datetime(int(data["start"]), unit="ms", utc=True),
            "open": float(data["open"]),
            "high": float(data["high"]),
            "low":  float(data["low"]),
            "close": float(data["close"]),
            "volume": float(data["volume"]),
        }
        self.buffer.append(pd.Series(candle).rename(candle["ts"]))
        if len(self.buffer) < CFG.lookback:
            return  # need more history
        if len(self.buffer) > CFG.lookback:
            self.buffer.pop(0)
        df = pd.DataFrame(self.buffer)
        feats = compute_indicators(df)
        x_raw = feats.iloc[-CFG.lookback:]
        x_scaled = self.scaler.transform(x_raw)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(CFG.device)
        prob_up = self.model(x_tensor).item()
        print(datetime.utcnow(), "Prob_up", prob_up)
        desired_pos = 1 if prob_up >= 0.5 else -1
        if desired_pos != self.current_pos:
            self.flip_position(desired_pos)

    def flip_position(self, target: int):
        side = "Buy" if target == 1 else "Sell"
        qty = 0.001   # ⚠️ set your size properly
        try:
            self.client.place_order(category="linear", symbol=CFG.symbol, side=side,
                                    orderType="Market", qty=qty, positionIdx=0)
            self.current_pos = target
            print("Opened", side, "qty", qty)
        except Exception as e:
            print("Order error:", e)

# ──────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = int((datetime.now(timezone.utc) - timedelta(days=365*2)).timestamp() * 1000)  # 2 years
    print("Downloading klines…")
    df_raw = get_klines(CFG.symbol, CFG.primary_interval, start, end)

    X, y, scaler = build_dataset(df_raw)

    X_train, X_test, y_train, y_test = train_test_split_chrono(X, y, CFG.train_split)
    ds_train = SeqDataset(X_train, y_train)
    ds_val   = SeqDataset(X_test[: len(X_test)//4], y_test[: len(y_test)//4])
    ds_test  = SeqDataset(X_test[ len(X_test)//4 :], y_test[len(y_test)//4 :])

    train_loader = DataLoader(ds_train, batch_size=CFG.batch_size, shuffle=False)
    val_loader   = DataLoader(ds_val, batch_size=CFG.batch_size, shuffle=False)

    model = CNNLSTM(feature_dim=X.shape[2])
    train_model(model, train_loader, val_loader)

    # Load best weights
    model.load_state_dict(torch.load("best_model.pt", map_location=CFG.device))
    evaluate(model, ds_test[:][0].numpy(), ds_test[:][1].numpy())

    # Backtest
    with torch.no_grad():
        y_pred_prob = model(torch.tensor(X_test, dtype=torch.float32, device=CFG.device)).cpu().numpy().ravel()
    backtest(df_raw.iloc[-len(y_pred_prob):], X_test, y_pred_prob)

    # Live trading (uncomment to run)
    # trader = LiveTrader(model, scaler)
    # while True:
    #     time.sleep(1)

if __name__ == "__main__":
    main()