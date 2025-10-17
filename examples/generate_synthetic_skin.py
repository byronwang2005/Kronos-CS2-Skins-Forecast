import os
import numpy as np
import pandas as pd

def generate_synthetic_skin_data(days,seed):
    np.random.seed(seed)
    
    # 时间戳：从 2021-01-01 开始的连续交易日
    timestamps = pd.date_range("2021-01-01", periods=days, freq="D")
    
    # 初始价格（例如：$10 的皮肤）
    price = 10.0
    prices = [price]
    
    # 模拟价格：随机游走 + 小幅趋势 + 噪声
    for _ in range(1, days):
        # 日收益率：均值 0.00%（微涨趋势），标准差 2%
        daily_return = np.random.normal(loc=0.0000, scale=0.02)
        price *= (1 + daily_return)
        price = max(price, 0.1)  # 价格不能为负或过低
        prices.append(price)
    
    # 模拟成交量：泊松分布（平均每天 50 笔交易）
    volume = np.random.poisson(lam=50, size=days).astype(float)
    
    # 成交额 = 价格 × 成交量
    amount = np.array(prices) * volume

    # 构建完整 DataFrame
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
    full_path = "examples/synthetic_skin_full.csv"
    df_full.to_csv(full_path, index=False)
    
    # 保存基础版（仅 OHLC）
    ohlc_cols = ["timestamps", "open", "high", "low", "close"]
    df_ohlc = df_full[ohlc_cols]
    ohlc_path = "examples/synthetic_skin_data.csv"
    df_ohlc.to_csv(ohlc_path, index=False)
    
    print("✅ 合成数据生成成功！")
    print(f"   - 基础版（OHLC）: {ohlc_path}")
    print(f"   - 完整版（OHLC+volume+amount）: {full_path}")
    print("\n💡 提示：这些数据仅用于演示，不包含任何真实市场信息。")

if __name__ == "__main__":
    main()