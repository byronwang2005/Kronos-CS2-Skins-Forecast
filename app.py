# demo/app.py
import os
import sys
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# 添加项目根目录到 sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

# 导入你的预测器（请确保 src/predictor.py 存在且可导入）
from src.predictor import CS2SkinPredictor

# 生成默认合成数据（避免依赖外部文件）
def generate_default_skin_data():
    from examples.generate_synthetic_skin import generate_synthetic_skin
    df = generate_synthetic_skin(
        start_date="2024-01-01",
        days=200,
        base_price=10.0,
        volatility=0.02,
        trend=0.0001,
        ohlcva=False
    )
    return df

# 预测函数
def forecast_skin_price(file, pred_days):
    try:
        if file is not None:
            # 用户上传 CSV
            df = pd.read_csv(file.name)
            required_cols = {'timestamps', 'open', 'high', 'low', 'close'}
            if not required_cols.issubset(df.columns):
                return None, "❌ 上传的 CSV 缺少必要列: timestamps, open, high, low, close"
        else:
            # 使用默认合成数据
            df = generate_default_skin_data()

        # 确保 timestamps 是 datetime
        df['timestamps'] = pd.to_datetime(df['timestamps'])

        # 初始化预测器（自动选设备）
        predictor = CS2SkinPredictor()

        # 预测
        pred_df = predictor.predict(df, pred_days=pred_days)

        # 绘图
        hist_len = min(3 * pred_days, len(df))
        hist_df = df.iloc[-hist_len:]
        hist_time = hist_df['timestamps']
        hist_close = hist_df['close']
        pred_time = pred_df.index
        pred_close = pred_df['close']

        plt.figure(figsize=(10, 4))
        plt.plot(hist_time, hist_close, label=f'Historical ({hist_len} days)', color='steelblue')
        plt.plot(pred_time, pred_close, label=f'Forecast ({pred_days} days)', color='crimson', linestyle='--')
        plt.title('CS2 Skin Price Forecast (Kronos)')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(rotation=30)
        plt.tight_layout()

        # 转为 base64 图像
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # 预测结果表格
        result_df = pred_df[['close']].round(4)
        return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%">', result_df.to_html()

    except Exception as e:
        return None, f"❌ 预测出错: {str(e)}"


# Gradio 界面
with gr.Blocks(title="Kronos CS2 Skin Forecast") as demo:
    gr.Markdown("# 🔮 Kronos CS2 皮肤价格预测 Demo")
    gr.Markdown("""
    - 使用 **Kronos-small**（金融时序大模型）预测 CS2 皮肤未来价格
    - 默认使用**合成数据**（合规！）
    - 你可上传自己的 OHLC CSV（格式：`timestamps, open, high, low, close`）
    - **本 Demo 不存储任何用户数据**
    """)

    with gr.Row():
        file_input = gr.File(label="上传你的皮肤价格 CSV（可选）", file_types=[".csv"])
        pred_days = gr.Slider(1, 14, value=7, step=1, label="预测天数")

    run_btn = gr.Button("🚀 开始预测")

    output_image = gr.HTML(label="预测结果图表")
    output_table = gr.HTML(label="预测价格（USD）")

    run_btn.click(
        fn=forecast_skin_price,
        inputs=[file_input, pred_days],
        outputs=[output_image, output_table]
    )

    gr.Markdown("### 📌 注意")
    gr.Markdown("""
    - 本项目 **不提供真实皮肤数据**，所有预测仅用于研究与教育
    - 模型基于 [Kronos](https://github.com/shiyu-coder/Kronos)（MIT 许可）
    - 作者与 Valve / Steam 无任何关联
    """)

if __name__ == "__main__":
    demo.launch()