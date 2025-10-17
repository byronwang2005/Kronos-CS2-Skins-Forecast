import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# 添加项目根目录以导入 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor import CS2SkinPredictor


def plot_prediction(hist_df, pred_df, pred_len):
    hist_close = hist_df['close'].iloc[-pred_len:]
    hist_time = hist_df['timestamps'].iloc[-pred_len:]
    pred_close = pred_df['close']
    pred_time = pred_df.index

    plt.figure(figsize=(12, 5))
    plt.plot(hist_time, hist_close, label=f'Historical Price (The Last {pred_len} Days)', color='steelblue', linewidth=2)
    plt.plot(pred_time, pred_close, label=f'Forecast Price (The Next {pred_len} days)', color='crimson', linestyle='--', linewidth=2)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.title('Forecast of CS2 Skin Prices and Trading Volumes(OHLC) - Kronos', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# === 加载数据 ===
data_path = "examples/synthetic_skin_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError("请先运行: python examples/generate_synthetic_skin.py")

df = pd.read_csv(data_path)
df['timestamps'] = pd.to_datetime(df['timestamps'])

# === 使用封装预测器 ===
print("🚀 初始化 CS2 皮肤价格预测器...")
predictor = CS2SkinPredictor()  # 自动加载 Kronos-small，自动选设备

print("🧠 运行预测...")
pred_df = predictor.predict(df, pred_days=7)

# === 输出与绘图 ===
print("\n📊 预测结果:")
print(pred_df[['close']])
plot_prediction(df, pred_df, pred_len=7)