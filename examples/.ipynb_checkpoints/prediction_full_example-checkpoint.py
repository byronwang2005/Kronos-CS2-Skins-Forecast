import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# 添加项目根目录以导入 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor import CS2SkinPredictor


def plot_prediction_full(hist_df, pred_df, pred_len):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 价格
    hist_close = hist_df['close'].iloc[-pred_len:]
    hist_time = hist_df['timestamps'].iloc[-pred_len:]
    pred_close = pred_df['close']
    pred_time = pred_df.index

    ax1.plot(hist_time, hist_close, label='Historical Price', color='steelblue', linewidth=2)
    ax1.plot(pred_time, pred_close, label='Forecast Price', color='crimson', linestyle='--', linewidth=2)
    ax1.set_ylabel('Price (USD)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.5)

    # 成交量（次坐标轴）
    ax2 = ax1.twinx()
    hist_vol = hist_df['volume'].iloc[-pred_len:]
    pred_vol = pred_df['volume']
    ax2.bar(hist_time, hist_vol, width=0.8, alpha=0.3, color='gray', label='Historical Trading Volume')
    ax2.bar(pred_time, pred_vol, width=0.8, alpha=0.6, color='orange', label='Forecast Trading Volume')
    ax2.set_ylabel('Trading Volume', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    plt.title('Forecast of CS2 Skin Prices and Trading Volumes(OHLCVA) - Kronos', fontsize=14)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# === 加载完整数据 ===
data_path = "examples/synthetic_skin_full.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError("请先运行: python examples/generate_synthetic_skin.py")

df = pd.read_csv(data_path)
df['timestamps'] = pd.to_datetime(df['timestamps'])

# === 使用封装预测器 ===
print("🚀 初始化 CS2 皮肤价格预测器（完整字段）...")
predictor = CS2SkinPredictor()  # 自动加载 Kronos-small，自动选设备

print("🧠 运行预测...")
pred_df = predictor.predict(df, pred_days=7)

# === 输出与绘图 ===
print("\n📊 预测结果（价格 + 成交量）:")
print(pred_df[['close', 'volume']])
plot_prediction_full(df, pred_df, pred_len=7)