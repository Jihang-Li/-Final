import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_mpl_pair_grid_style_beta_eq_gamma(data: dict, param_dict: dict, fig_folder="results", show=True):
    """
    使用多个 beta=gamma 的参数组对多个数据文件进行预测，输出 Markdown 表格和九宫格拟合图。

    参数：
        data (dict): 数据集，键为文件名，值为包含 "step", "lrs", "loss" 的字典。
        param_dict (dict): 参数组字典，键为模型名，值为 [L0, A, alpha, B, C, beta]。
        fig_folder (str): 保存图像的路径。
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
        L0, A, alpha, B, C, beta = param_dict[label]
        gamma = beta  # 强制令 gamma = beta

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
    print("\n📊 Markdown 表格（beta = gamma）\n")
    print("| Using Params | Predict Target | MSE     | R²      | MAE     | MAPE    | MaxPE   | RMSE    |")
    print("|--------------|----------------|---------|---------|---------|---------|---------|---------|")
    for r in results:
        print(f"| {r['Using']:<12} | {r['Predict']:<14} | {r['MSE']:.6f} | {r['R2']:.6f} | {r['MAE']:.6f} | "
              f"{r['MAPE']:.6f} | {r['MaxPE']:.6f} | {r['RMSE']:.6f} |")

    fig.tight_layout()
    save_path = os.path.join(fig_folder, "grid_fit_beta_eq_gamma.png")
    plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

param_dict = {
    "811":   [0.1146119395553969, 2.5960129316926657, 0.03296413140723903, 110.07174779462125, 2.90312534743649, 0.7831220115990577],
    "wsd": [0.18240186658119067, 2.5287328841410113, 0.03294743683621894, 116.55743462558041, 3.04623577199019, 0.8477446773846402],
    "cosine": [0.21163331162300478, 2.4958942972292113, 0.03444932849973524, 165.95441258692378, 1.9955073086964734, 0.6994383103788475],
}

evaluate_mpl_pair_grid_style_beta_eq_gamma(
    data, param_dict,
    fig_folder="results_crossfit_2500_50_beta_eq_gamma",
    show=True
)
