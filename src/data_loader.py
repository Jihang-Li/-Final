import os
import pandas as pd

def load_data(folder_path):
    """
    自动读取指定目录中所有 .csv/.xls/.xlsx 文件（包含 step, lr, loss 字段），返回结构化数据。
    """
    data = {}
    for file_name in os.listdir(folder_path):
        if not (file_name.endswith(".csv") or file_name.endswith(".xls") or file_name.endswith(".xlsx")):
            continue

        file_path = os.path.join(folder_path, file_name)

        try:
            if file_name.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        except Exception as e:
            print(f"❌ 读取失败: {file_name} ({e})")
            continue

        # 标准化列名
        df = df.rename(columns={
            "Metrics/loss": "loss",
            "lrs": "lr",
            "learning_rate": "lr",
            "Learning Rate": "lr"
        })

        if not {"step", "lr", "loss"}.issubset(df.columns):
            print(f"❌ 缺少列: {file_name}，实际列: {df.columns.tolist()}")
            continue

        data[file_name] = {
            "step": df["step"].values,
            "lr": df["lr"].values,
            "loss": df["loss"].values
        }

    return data
