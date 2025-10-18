import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# 添加项目根目录以导入 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor import CS2SkinPredictor


def plot_prediction_full_for_skin(hist_df, pred_df, skin_id, pred_len):
    """为单个皮肤绘制价格+成交量预测图"""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 价格
    hist_close = hist_df['close'].iloc[-9*pred_len:]
    hist_time = hist_df['timestamps'].iloc[-9*pred_len:]
    pred_close = pred_df['close']
    pred_time = pred_df['timestamps']

    ax1.plot(hist_time, hist_close, label='Historical Price', color='steelblue', linewidth=2)
    ax1.plot(pred_time, pred_close, label='Forecast Price', color='crimson', linestyle='--', linewidth=2)
    ax1.set_ylabel('Price (USD)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.5)

    # 成交量（次坐标轴）
    ax2 = ax1.twinx()
    hist_vol = hist_df['volume'].iloc[-9*pred_len:]
    pred_vol = pred_df['volume']
    ax2.bar(hist_time, hist_vol, width=0.8, alpha=0.3, color='gray', label='Historical Volume')
    ax2.bar(pred_time, pred_vol, width=0.8, alpha=0.6, color='orange', label='Forecast Volume')
    ax2.set_ylabel('Trading Volume', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    plt.title(f'CS2 Skin Forecast (OHLCVA) - {skin_id} | Kronos', fontsize=14)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# === 配置 ===
PRED_DAYS = 7
NUM_PLOTS = 3  # 只可视化前 N 个皮肤

# 尝试加载批量完整数据
batch_full_path = "examples/data/synthetic_500_skins_full.csv"
single_full_path = "examples/data/synthetic_skin_full.csv"

if os.path.exists(batch_full_path):
    print("📁 检测到批量完整数据（含 volume/amount），进行批量预测...")
    df = pd.read_csv(batch_full_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    use_batch = True
elif os.path.exists(single_full_path):
    print("⚠️ 未找到批量完整数据，回退到单皮肤完整预测...")
    df = pd.read_csv(single_full_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    use_batch = False
else:
    raise FileNotFoundError(
        "请先生成数据:\n"
        "  python examples/generate_synthetic_skin.py          # 单皮肤\n"
        "或\n"
        "  python examples/generate_500_skins.py               # 批量皮肤（需含 volume/amount）"
    )

# === 初始化预测器 ===
print("🚀 初始化 CS2 皮肤价格预测器（OHLCVA）...")
predictor = CS2SkinPredictor()

# === 执行预测 ===
if use_batch:
    print(f"🧠 批量预测 {df['skin_id'].nunique()} 个皮肤（含成交量），未来 {PRED_DAYS} 天...")
    pred_results = predictor.predict_batch(
        df,
        skin_id_col="skin_id",
        pred_days=PRED_DAYS,
        T=0.8,
        top_p=0.9
    )
    print("✅ 批量预测完成！")
    
    # 保存结果
    os.makedirs("examples/data", exist_ok=True)
    pred_results.to_csv("examples/data/predictions_500_skins_full.csv", index=False)
    print("💾 预测结果已保存至: examples/data/predictions_500_skins_full.csv")

    # 可视化前几个皮肤
    for skin_id in pred_results['skin_id'].unique()[:NUM_PLOTS]:
        hist = df[df['skin_id'] == skin_id]
        pred = pred_results[pred_results['skin_id'] == skin_id]
        plot_prediction_full_for_skin(hist, pred, skin_id, PRED_DAYS)

else:
    print("🧠 单皮肤完整字段预测...")
    pred_df = predictor.predict(df, pred_days=PRED_DAYS)
    print("\n📊 预测结果（价格 + 成交量）:")
    print(pred_df[['close', 'volume']])
    # 准备绘图格式
    pred_df_plot = pred_df.copy()
    pred_df_plot['timestamps'] = pred_df_plot.index
    plot_prediction_full_for_skin(df, pred_df_plot, "Single Skin", PRED_DAYS)