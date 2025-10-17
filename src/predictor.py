import os
import sys
import torch
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Kronos, KronosTokenizer, KronosPredictor


class CS2SkinPredictor:
    """
    封装 Kronos 模型，用于 CS2 皮肤价格预测。
    支持仅 OHLC 或完整 OHLCV 输入。
    """

    def __init__(self, model_name="NeoQuasar/Kronos-small", tokenizer_name="NeoQuasar/Kronos-Tokenizer-base"):
        """
        初始化预测器。
        
        Args:
            model_name (str): Hugging Face 上的 Kronos 模型名称
            tokenizer_name (str): 对应的 Tokenizer 名称
        """
        self.device = self._get_device()
        print(f"✅ 使用设备: {self.device}")

        print(f"📥 加载 Tokenizer: {tokenizer_name}")
        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)

        print(f"📥 加载模型: {model_name}")
        self.model = Kronos.from_pretrained(model_name)

        # Kronos-small / base 的 max_context 为 512
        self.predictor = KronosPredictor(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_context=512
        )

    def _get_device(self):
        """自动选择可用设备"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def predict(self, df: pd.DataFrame, pred_days: int = 7, T: float = 0.8, top_p: float = 0.9):
        """
        对皮肤价格进行预测。
        
        Args:
            df (pd.DataFrame): 历史数据，必须包含 'timestamps', 'open', 'high', 'low', 'close'
                               可选包含 'volume', 'amount'
            pred_days (int): 预测未来天数（1–14）
            T (float): 采样温度
            top_p (float): 核采样概率
            
        Returns:
            pd.DataFrame: 预测结果，索引为未来日期，包含 OHLC(VA)
        """
        # 输入校验
        required_cols = ["timestamps", "open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"输入数据必须包含列: {required_cols}")

        df = df.copy()
        df["timestamps"] = pd.to_datetime(df["timestamps"])

        # 自动选择字段（兼容有无 volume/amount）
        available_cols = ["open", "high", "low", "close"]
        if "volume" in df.columns and "amount" in df.columns:
            available_cols += ["volume", "amount"]
        elif "volume" in df.columns or "amount" in df.columns:
            print("仅提供 volume 或 amount 中的一个，将忽略该字段。建议同时提供或都不提供。")
        x_df = df[available_cols].iloc[-400:]  # 不超过 max_context=512
        x_timestamp = df["timestamps"].iloc[-400:]

        # 生成未来时间戳（必须为 pd.Series）
        y_timestamp = pd.Series(
            pd.date_range(
                start=x_timestamp.iloc[-1] + pd.Timedelta(days=1),
                periods=pred_days,
                freq="D"
            )
        )

        # 执行预测
        pred_df = self.predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_days,
            T=T,
            top_p=top_p,
            sample_count=1,
            verbose=False
        )
        pred_df.index = y_timestamp
        return pred_df