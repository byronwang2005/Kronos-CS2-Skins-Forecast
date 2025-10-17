import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Kronos, KronosTokenizer, KronosPredictor


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

    plt.title('Forecast of CS2 Skin Prices and Trading Volumes(Full) - Kronos', fontsize=14)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# === 设备检测 ===
device = "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ 使用设备: {device}")

# === 加载模型 ===
print("📥 加载 Kronos-small（完整字段）...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

# === 加载完整数据 ===
data_path = "examples/synthetic_skin_full.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError("请先运行: python examples/generate_synthetic_skin.py")

df = pd.read_csv(data_path)
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = min(400, len(df))
pred_len = 7

# 包含 volume 和 amount
x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.iloc[-lookback:]['timestamps']
y_timestamp = pd.Series(pd.date_range(
    start=x_timestamp.iloc[-1] + pd.Timedelta(days=1),
    periods=pred_len,
    freq='D'
))

# === 预测 ===
print("🧠 运行完整字段预测...")
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=0.8,
    top_p=0.9,
    sample_count=1,
    verbose=True
)
pred_df.index = y_timestamp

# === 输出与绘图 ===
print("\n📊 预测结果（价格 + 成交量）:")
print(pred_df[['close', 'volume']])
plot_prediction_full(df, pred_df, pred_len)