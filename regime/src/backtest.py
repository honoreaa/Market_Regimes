import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest_strategy(df: pd.DataFrame, regime_label_col: str = "regime", trend_label=0):
    df["signal"] = 0
    df.loc[df[regime_label_col] == trend_label, "signal"] = 1
    df["strategy_return"] = df["signal"].shift(1) * df["return"]
    df.dropna(inplace=True)

    cumulative = (1 + df[["return", "strategy_return"]]).cumprod()
    fig, ax = plt.subplots(figsize=(14,6))
    cumulative.plot(ax=ax, title="Regime-Aware Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.show()
    return df, cumulative

def sharpe_ratio(series, risk_free=0):
    excess = series - risk_free/252
    return (excess.mean() / excess.std()) * np.sqrt(252)

def max_drawdown(cum_returns):
    rolling_max = cum_returns.cummax()
    drawdown = cum_returns / rolling_max - 1
    return drawdown.min()
