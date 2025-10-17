import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(hist_df, pred_df, pred_len):
    hist_close = hist_df['close'].iloc[-pred_len:]
    hist_time = hist_df['timestamps'].iloc[-pred_len:]
    pred_close = pred_df['close']
    pred_time = pred_df.index

    plt.figure(figsize=(12, 5))
    plt.plot(hist_time, hist_close, label=f'Historical Price (The Last {pred_len} Days)', color='steelblue', linewidth=2)
    plt.plot(pred_time, pred_close, label=f'Forecast Price (The Next {pred_len} days)', color='crimson', linestyle='--', linewidth=2)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.title('Forecast of CS2 Skin Prices and Trading Volumes - Kronos', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# === 设备检测 ===
device = "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ 使用设备: {device}")

# === 加载模型 ===
print("📥 加载 Kronos-small...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

# === 加载数据 ===
data_path = "examples/synthetic_skin_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError("请先运行: python examples/generate_synthetic_skin.py")

df = pd.read_csv(data_path)
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = min(400, len(df))
pred_len = 7

x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close']]
x_timestamp = df.iloc[-lookback:]['timestamps']
y_timestamp = pd.Series(pd.date_range(
    start=x_timestamp.iloc[-1] + pd.Timedelta(days=1),
    periods=pred_len,
    freq='D'
))

# === 预测 ===
print("🧠 运行预测...")
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
print("\n📊 预测结果（收盘价）:")
print(pred_df[['close']])
plot_prediction(df, pred_df, pred_len)