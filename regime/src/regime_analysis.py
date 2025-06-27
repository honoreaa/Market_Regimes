import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def max_drawdown_calc(series):
    """
    Calculate maximum drawdown from a price series.
    """
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    max_dd = drawdown.min()
    return max_dd

def regime_statistics(df):
    """
    Calculate and print key statistics for each regime.
    Also return a DataFrame summary.
    """
    stats = []

    for reg in sorted(df['regime'].unique()):
        subset = df[df['regime'] == reg]
        mean_return = subset['return'].mean()
        vol = subset['return'].std()
        sharpe = mean_return / vol if vol != 0 else 0
        max_drawdown = max_drawdown_calc(subset['Close'])
        duration = len(subset)

        stats.append({
            'Regime': reg,
            'Mean Return': mean_return,
            'Volatility': vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Duration (days)': duration
        })

    stats_df = pd.DataFrame(stats)
    print(stats_df)
    return stats_df

def simple_regime_backtest(df, long_regimes):
    """
    Simulate a simple strategy: long only during regimes in long_regimes list, flat otherwise.
    Assumes df has 'return' and 'regime' columns.
    
    Returns df with strategy returns and cumulative returns.
    """
    df = df.copy()
    df['position'] = np.where(df['regime'].isin(long_regimes), 1, 0)
    df['strategy_return'] = df['position'] * df['return']
    df['market_cumret'] = (1 + df['return']).cumprod() - 1
    df['strategy_cumret'] = (1 + df['strategy_return']).cumprod() - 1
    return df

def plot_regimes(df, save_path=None):
    plt.figure(figsize=(15,7))
    regimes = sorted(df['regime'].unique())
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']  # Extended palette

    for i, reg in enumerate(regimes):
        subset = df[df['regime'] == reg]
        plt.scatter(subset.index, subset['Close'], label=f"Regime {reg}", c=colors[i % len(colors)], s=5)

    plt.legend()
    plt.title("Market Regimes Detected by HMM")
    plt.xlabel("Date")
    plt.ylabel("Price")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_regime_statistics(stats_df, save_path=None):
    """
    Barplot of regime stats: mean return, volatility, sharpe, max drawdown.
    """
    plt.figure(figsize=(14, 8))
    stats_df_melted = stats_df.melt(id_vars='Regime', var_name='Metric', value_name='Value')

    sns.barplot(data=stats_df_melted, x='Regime', y='Value', hue='Metric')
    plt.title("Regime Statistics Comparison")
    plt.ylabel("Value")
    plt.xlabel("Regime")
    plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_strategy_vs_market(df, save_path=None):
    """
    Plot cumulative returns of market vs strategy.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['market_cumret'], label='Market (Buy & Hold)', color='blue')
    plt.plot(df.index, df['strategy_cumret'], label='Strategy (Regime Based)', color='green')
    plt.title("Cumulative Returns: Market vs Regime-Based Strategy")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()
