import os
import numpy as np
import pandas as pd

def generate_synthetic_skin_data(days,seed):
    np.random.seed(seed)
    timestamps = pd.date_range("2021-01-01", periods=days, freq="D")
    
    price = 10.0
    prices = [price]
    volume_base = 50  # 基础成交量

    for t in range(1, days):
        # 1. 长期趋势：年化 ~30% 增长（皮肤市场的“信仰溢价”）
        trend = 0.0008  # ≈ 30% 年化 (0.0008 * 252 ≈ 0.2)

        # 2. 波动率：平时 1.5%，但每 100 天左右有一次“热度事件”（波动↑）
        if t % 100 < 5:  # 模拟短期事件（赛事、箱子停售）
            vol = 0.04
            event_boost = np.random.uniform(0.02, 0.08)  # 额外上涨
        else:
            vol = 0.015
            event_boost = 0.0

        # 3. 收益率 = 趋势 + 事件 + 噪声
        daily_return = trend + event_boost + np.random.normal(0, vol)

        # 4. 偶尔崩盘（每 300 天左右 5% 概率大跌）
        if t > 100 and np.random.rand() < 0.005:  # 0.5% 日概率
            daily_return -= np.random.uniform(0.1, 0.3)  # 跌 10%~30%

        price *= (1 + daily_return)
        price = max(price, 0.1)
        prices.append(price)

    # 成交量：价格越高，成交量越大（正反馈）
    volume = np.random.poisson(lam=volume_base + np.array(prices) * 5, size=days).astype(float)
    amount = np.array(prices) * volume

    df = pd.DataFrame({
        "timestamps": timestamps,
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": volume,
        "amount": amount
    })
    return df


def main():
    # 确保 examples/ 目录存在
    os.makedirs("examples", exist_ok=True)
    
    # 生成数据
    df_full = generate_synthetic_skin_data(days=1000, seed=42)
    
    # 保存完整版（含 volume/amount）
    full_path = "examples/data/synthetic_skin_full.csv"
    df_full.to_csv(full_path, index=False)
    
    # 保存基础版（仅 OHLC）
    ohlc_cols = ["timestamps", "open", "high", "low", "close"]
    df_ohlc = df_full[ohlc_cols]
    ohlc_path = "examples/data/synthetic_skin_ohlc.csv"
    df_ohlc.to_csv(ohlc_path, index=False)
    
    print("✅ 合成数据生成成功！")
    print(f"   - 基础版（OHLC）: {ohlc_path}")
    print(f"   - 完整版（OHLC+volume+amount）: {full_path}")
    print("\n💡 提示：这些数据仅用于演示，不包含任何真实市场信息。")

if __name__ == "__main__":
    main()