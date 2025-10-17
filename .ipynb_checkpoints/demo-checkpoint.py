import pandas as pd
from src.predictor import CS2SkinPredictor
import matplotlib.pyplot as plt

# 1. 加载数据（可以是合成数据，也可以是用户自己的合规数据）
df = pd.read_csv("examples/synthetic_skin_full.csv")

# 2. 初始化预测器（自动选设备，加载 Kronos-small）
predictor = CS2SkinPredictor()

# 3. 预测未来 7 天
result = predictor.predict(df, pred_days=7)

# 4. 查看结果
print("预测的收盘价：")
print(result[["close"]])

# 5. 可选：绘图
result[["close"]].plot(title="CS2", grid=True)
plt.show()