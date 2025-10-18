import os
import numpy as np
import pandas as pd

def generate_one_skin(days: int, seed: int, base_price: float = 10.0):
    np.random.seed(seed)
    timestamps = pd.date_range("2021-01-01", periods=days, freq="D")
    
    price = base_price
    prices = [price]
    
    # 更贴近 CS 皮肤的动态：长期微涨 + 事件驱动 + 偶尔崩盘
    for t in range(1, days):
        # 基础趋势：年化 ~15-30%
        trend = np.random.uniform(0.0005, 0.001)  # 日趋势随机化，避免同质
        
        # 波动率
        vol = 0.02
        
        # 事件：每 80~120 天一次热度事件（持续 3 天）
        if any((t - offset) % np.random.randint(80, 120) == 0 for offset in range(3)):
            event_return = np.random.uniform(0.03, 0.1)  # 单日涨 3%~10%
        else:
            event_return = 0.0

        # 崩盘风险：低概率大跌
        crash = -np.random.uniform(0.1, 0.4) if np.random.rand() < 0.002 else 0.0

        daily_return = trend + event_return + crash + np.random.normal(0, vol)
        price *= (1 + daily_return)
        price = max(price, 0.1)
        prices.append(price)

    volume = np.random.poisson(lam=30 + np.array(prices) * 3, size=days).astype(float)
    amount = np.array(prices) * volume

    return pd.DataFrame({
        "timestamps": timestamps,
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": volume,
        "amount": amount
    })

def main():
    os.makedirs("examples/data", exist_ok=True)
    
    n_skins = 500
    days = 1000
    all_skins = []

    print(f"🔄 正在生成 {n_skins} 个皮肤的合成数据...")
    for i in range(n_skins):
        base_price = np.random.uniform(5, 50)  # 初始价格 5~50 美元
        df_skin = generate_one_skin(days=days, seed=42 + i, base_price=base_price)
        df_skin["skin_id"] = f"skin_{i:03d}"
        all_skins.append(df_skin)

    df_all = pd.concat(all_skins, ignore_index=True)
    
    # 保存完整长表
    df_all.to_csv("examples/data/synthetic_500_skins_full.csv", index=False)
    
    # 保存仅 OHLC + skin_id（Kronos 所需最小集）
    ohlc_cols = ["skin_id", "timestamps", "open", "high", "low", "close"]
    df_all[ohlc_cols].to_csv("examples/data/synthetic_500_skins_ohlc.csv", index=False)

    print("✅ 500 个皮肤数据生成成功！")
    print("   - 完整版: examples/data/synthetic_500_skins_full.csv")
    print("   - OHLC版: examples/data/synthetic_500_skins_ohlc.csv")
    print("\n💡 提示：可用此数据测试 Kronos 的 batch 预测能力。")

if __name__ == "__main__":
    main()