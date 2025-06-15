import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_mpl_pair_grid_style(data: dict, param_dict: dict, fig_folder="results", show=True):
    """
    使用多个参数组对多个数据文件进行预测，输出 Markdown 表格和九宫格拟合图。

    参数：
        data (dict): 训练数据，键为文件名，值为包含 "step", "lrs", "loss" 的字典。
        param_dict (dict): 参数组字典，键为模型标签，值为 7 个 float 参数。
        fig_folder (str): 图像保存路径。
        show (bool): 是否显示图像。
    """
    os.makedirs(fig_folder, exist_ok=True)

    labels = list(param_dict.keys())
    files = list(data.keys())

    n_row = len(labels)
    n_col = len(files)
    fig, axs = plt.subplots(n_row, n_col, figsize=(4 * n_col, 3.5 * n_row))
    results = []

    for i, label in enumerate(labels):
        L0, A, alpha, B, C, beta, gamma = param_dict[label]
        for j, file_name in enumerate(files):
            d = data[file_name]
            lrs = d["lrs"]
            step = d["step"]
            loss = d["loss"]
            lr_sum = np.cumsum(lrs)
            lr_gap = np.zeros_like(lrs)
            lr_gap[1:] = np.diff(lrs)

            S1 = lr_sum[step]
            LD = np.zeros_like(step, dtype=np.float64)

            for k, s in enumerate(step):
                if s < 1:
                    continue
                lr_seg = lrs[1:s+1]
                s1_gap = lr_sum[s] - lr_sum[:s]
                term = 1 + C * lr_seg**(-gamma) * s1_gap
                LD[k] = np.sum(lr_gap[1:s+1] * (1 - term**(-beta)))

            pred = L0 + A * S1**(-alpha) + B * LD

            mse = mean_squared_error(loss, pred)
            r2 = r2_score(loss, pred)
            mae = np.mean(np.abs(loss - pred))
            mape = np.mean(np.abs(loss - pred) / loss)
            maxpe = np.max(np.abs(loss - pred) / loss)
            rmse = np.sqrt(mse)

            results.append({
                "Using": label,
                "Predict": file_name,
                "MSE": mse,
                "R2": r2,
                "MAE": mae,
                "MAPE": mape,
                "MaxPE": maxpe,
                "RMSE": rmse,
            })

            ax = axs[i][j] if n_row > 1 else axs[j]
            ax.plot(step, loss, 'o', label="True", markersize=2)
            ax.plot(step, pred, '-', label="Pred", linewidth=2)
            ax.set_title(f"{label} → {file_name}", fontsize=10)
            ax.grid(True)
            ax.legend(fontsize=8)

    # 表格输出
    print("\n📊 Markdown 表格：\n")
    print("| Using Params | Predict Target | MSE     | R²      | MAE     | MAPE    | MaxPE   | RMSE    |")
    print("|--------------|----------------|---------|---------|---------|---------|---------|---------|")
    for r in results:
        print(f"| {r['Using']:<12} | {r['Predict']:<14} | {r['MSE']:.6f} | {r['R2']:.6f} | {r['MAE']:.6f} | "
              f"{r['MAPE']:.6f} | {r['MaxPE']:.6f} | {r['RMSE']:.6f} |")

    fig.tight_layout()
    save_path = os.path.join(fig_folder, "grid_fit.png")
    plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()
import pickle

with open(r"C:\Users\lenovo\Desktop\processed_data_50_after2500.pkl", "rb") as f:
    data = pickle.load(f)

param_dict = {
    "811":  [0.1166083165880753, 2.5942307037606214, 0.03298865860769674, 110.00455053149541, 2.907638547154968, 0.8070187677255197,  0.775863088601702],
    "wsd":  [0.18240303435931915, 2.5287325251995925, 0.03294730607008809, 116.54713468275473, 3.0486705064389255, 0.8683428323219542,0.8385681586322705],
    "cosine": [0.2116682496664125, 2.4959150047295213, 0.034438041866009815, 165.94848290093566, 1.9985425949581603, 0.7160772863838875, 0.6904092350870443],
}

evaluate_mpl_pair_grid_style(data, param_dict, fig_folder="results_crossfit_2500_50", show=True)
