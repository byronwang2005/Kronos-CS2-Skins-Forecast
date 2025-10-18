import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# 添加项目根目录以导入 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor import CS2SkinPredictor


def plot_prediction_for_skin(hist_df, pred_df, skin_id, pred_len):
    """为单个皮肤绘制预测图"""
    hist_close = hist_df['close'].iloc[-9*pred_len:]
    hist_time = hist_df['timestamps'].iloc[-9*pred_len:]
    pred_close = pred_df['close']
    pred_time = pred_df['timestamps']

    plt.figure(figsize=(12, 5))
    plt.plot(hist_time, hist_close, label=f'Historical (Last {9*pred_len} Days)', color='steelblue', linewidth=2)
    plt.plot(pred_time, pred_close, label=f'Forecast (Next {pred_len} Days)', color='crimson', linestyle='--', linewidth=2)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.title(f'CS2 Skin Forecast (OHLC) - {skin_id} | Kronos', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# === 配置 ===
PRED_DAYS = 7
NUM_PLOTS = 3  # 只可视化前 N 个皮肤

# 尝试加载批量数据，若不存在则提示
batch_data_path = "examples/data/synthetic_500_skins_ohlc.csv"
single_data_path = "examples/data/synthetic_skin_data.csv"

if os.path.exists(batch_data_path):
    print("📁 检测到批量皮肤数据，将进行批量预测...")
    df = pd.read_csv(batch_data_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    use_batch = True
elif os.path.exists(single_data_path):
    print("⚠️ 未找到批量数据，回退到单皮肤预测...")
    df = pd.read_csv(single_data_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    use_batch = False
else:
    raise FileNotFoundError("请先运行:\n"
                            "  python examples/generate_synthetic_skin.py\n"
                            "或\n"
                            "  python examples/generate_500_skins.py")

# === 初始化预测器 ===
print("🚀 初始化 CS2 皮肤价格预测器（OHLC only）...")
predictor = CS2SkinPredictor()

# === 执行预测 ===
if use_batch:
    print(f"🧠 批量预测 {df['skin_id'].nunique()} 个皮肤，未来 {PRED_DAYS} 天...")
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
    pred_results.to_csv("examples/data/predictions_500_skins.csv", index=False)
    print("💾 预测结果已保存至: examples/data/predictions_500_skins.csv")

    # 可视化前几个皮肤
    for skin_id in pred_results['skin_id'].unique()[:NUM_PLOTS]:
        hist = df[df['skin_id'] == skin_id]
        pred = pred_results[pred_results['skin_id'] == skin_id]
        plot_prediction_for_skin(hist, pred, skin_id, PRED_DAYS)

else:
    print("🧠 单皮肤预测...")
    pred_df = predictor.predict(df, pred_days=PRED_DAYS)
    print("\n📊 预测结果:")
    print(pred_df[['close']])
    # 复用原绘图逻辑（稍作调整）
    pred_df_for_plot = pred_df.copy()
    pred_df_for_plot['timestamps'] = pred_df_for_plot.index
    plot_prediction_for_skin(df, pred_df_for_plot, "Single Skin", PRED_DAYS)