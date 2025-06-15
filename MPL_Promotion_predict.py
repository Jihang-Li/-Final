import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.special import huber
from torch import nn
import warnings
from scipy.optimize import minimize
from scipy.stats import linregress
from itertools import product
from tqdm import tqdm
import pickle

# Loss,过程量与绘图

def torch_huber(delta:float,r:torch.Tensor) -> torch.Tensor:
    '''基于torch对huber_loss 进行计算'''
    return torch.where(torch.abs(r) < delta, 0.5 * r ** 2, delta * (torch.abs(r) - 0.5 * delta))

def huber_loss(r:np.ndarray,delta:float=0.001) -> np.ndarray:
    """Compute Huber loss for numpy array"""
    return huber(delta,r)

def preprocess_data(data:dict, file_names:list) -> dict:
    torch_data = {}
    for file_name in file_names:
        torch_data[file_name] = {
            "step": torch.tensor(data[file_name]["step"], dtype=torch.int32),
            "lrs": torch.tensor(data[file_name]["lrs"], dtype=torch.float64),
            "loss": torch.tensor(data[file_name]["loss"], dtype=torch.float32),
        }
        lr_sum = torch.cumsum(torch_data[file_name]["lrs"], dim=0, dtype=torch.float64)
        torch_data[file_name]["S1"] = lr_sum[torch_data[file_name]["step"]]
        torch_data[file_name]["lr_sum"] = lr_sum
        lr_gap = torch.zeros_like(torch_data[file_name]["lrs"])
        lr_gap[1:] = torch.diff(torch_data[file_name]["lrs"])
        torch_data[file_name]["lr_gap"] = lr_gap
    return torch_data

def compute_loss(model,torch_data, train_set, optimizer):
    """计算全局损失，表现优化步骤"""
    optimizer.zero_grad()
    total_loss = 0.0
    for file_name in train_set:
        args = [torch_data[file_name][key] for key in ["S1", "lrs", "lr_sum", "step", "lr_gap", "loss"]]
        total_loss += model(*args)
    total_loss.backward()
    optimizer.step()
    return total_loss

def compute_grad_norm(model):
    """计算梯度的L2-范数"""
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    return torch.cat(grads).norm() if grads else torch.tensor(0.0)

def log_step(step, total_loss, best_loss, model, grad_norm):
    """
    记录训练进度并输出当前步骤的详细信息。
    
    参数说明：
        step (int)：当前训练步骤。
        total_loss (float)：当前步骤的损失值。
        best_loss (float)：迄今为止观察到的最优损失值。
        model (nn.Module)：MPL 模型实例。
        grad_norm (float)：当前步骤的梯度范数。
    """

    logger = logging.getLogger(__name__)
    params = {name: param.item() for name, param in model.named_parameters()}
    logger.info(f"Step {step:4d}: Loss={total_loss:.6f}, Best Loss={best_loss:.6f}, Grad Norm={grad_norm:.2e}")
    logger.info(f"Parameters: L0={params['L0']:.4f}, A={params['A']:.4f}, alpha={params['alpha']:.4f}, "
                f"B={params['B']:.4f}, C={params['C']:.4f}, beta={params['beta']:.4f}, gamma={params['beta']:.4f}")

def plot_loss_curve(loss_history, fig_folder):
    """Plot and save loss curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(loss_history)), loss_history, label="Fitting Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{fig_folder}/loss_monitor.png")
    plt.close()

class MPL(nn.Module):
    """
    基于学习率调度预测训练损失的多重幂律（MPL）模型（强制设定 gamma = beta）。

    参数说明：
        L0 (float): 基线损失参数。
        A (float): 幂律衰减项的幅值。
        alpha (float): 幂律衰减项的指数。
        B (float): 损失突降项的幅值。
        C (float): 损失突降变换中的缩放因子。
        beta (float): 幂律突降项中的指数（同时用于 lrs 的指数，即 gamma）。
    """

    def __init__(self, L0, A, alpha, B, C, beta):
        super().__init__()
        self.L0 = nn.Parameter(torch.tensor(L0, dtype=torch.float64))
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float64))
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float64))
        self.B = nn.Parameter(torch.tensor(B, dtype=torch.float64))
        self.C = nn.Parameter(torch.tensor(C, dtype=torch.float64))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float64))
        # gamma 被省略，模型内部统一用 beta 代替

    def forward(self, S1, lrs, lr_sum, step, lr_gap, loss):
        LD = torch.zeros_like(step, dtype=torch.float64)
        for i, s in enumerate(step):
            if s > 0:
                LD[i] = torch.sum(
                    lr_gap[1:s+1] * (
                        1 - (1 + self.C * lrs[1:s+1] ** (-self.beta) * (lr_sum[s] - lr_sum[:s])) ** (-self.beta)
                    )
                )
        pred = self.L0 + self.A * S1 ** (-self.alpha) + self.B * LD
        r = torch.log(loss) - torch.log(pred.clamp(min=1e-10))
        return torch_huber(0.001, r).sum()



from src.config import  FIT_MAX_STEPS,FIT_EVAL_INTERVAL, FIT_LR1, FIT_LR2, FIT_GRAD_NORM_THR, FIT_LOSS_THR, FIT_PATIENCE

def initialize_params(data: dict, train_set: list) -> list:
    """
    使用网格搜索和 L-BFGS-B 优化器初始化 MPL 模型的参数。

    参数：
        data (dict)：包含每个文件的步骤、学习率和损失值的数据集。
        train_set (list)：训练文件名列表。

    返回值：
        list：初始参数估计值 [L0, A, alpha, B]。
    """
    logger = logging.getLogger(__name__)
    logger.info("开始参数初始化")

    # 计算训练集中所有文件的最小损失值
    min_loss = min(data[file_name]["loss"].min() for file_name in train_set)
    log_y_list, log_x_list = [], []

    # 构造 log-log 回归所需的数据
    for file_name in train_set:
        log_y = np.log(data[file_name]["loss"] - min_loss + 0.01)
        log_x = np.log(np.cumsum(data[file_name]["lrs"])[data[file_name]["step"]])
        log_y_list.append(log_y)
        log_x_list.append(log_x)

    log_y = np.concatenate(log_y_list)
    log_x = np.concatenate(log_x_list)
    slope, intercept, _, _, _ = linregress(log_x, log_y)

    # 初始参数网格设定
    L0_init_set = np.linspace(min_loss - 0.2, min_loss + 0.2, 5)
    A_init_set = np.linspace(np.exp(intercept) - 0.1, np.exp(intercept) + 0.1, 3)
    alpha_init_set = np.linspace(-slope - 0.1, -slope + 0.1, 3)
    B_init_set = np.linspace(100, 1000, 3)

    # 拟合的目标损失函数（Huber Loss）
    def loss_fn0(params):
        L0, A, alpha, B = params
        total_loss = 0
        for file_name in train_set:
            lr = data[file_name]["lrs"]
            step = data[file_name]["step"]
            pred = L0 + A * np.cumsum(lr)[step] ** (-alpha) - B * (3e-4 - lr[step])
            loss = data[file_name]["loss"]
            r = np.log(loss) - np.log(pred)
            total_loss += huber_loss(r).sum()
        return total_loss

    init_params = list(product(L0_init_set, A_init_set, alpha_init_set, B_init_set))
    best_loss = float('inf')
    best_params = None

    # 遍历所有初始参数组合，寻找最优解
    for init_param in tqdm(init_params, desc="初始化参数搜索"):
        res = minimize(
            loss_fn0, init_param, method='L-BFGS-B', bounds=[(0, np.inf)] * 4,
            options={'maxiter': 100000, 'ftol': 1e-9, 'gtol': 1e-6, 'eps': 1e-8}
        )
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x

    logger.info(f"初始化完成。最优损失: {best_loss}, 最优参数: {best_params}")
    return best_params
def generate_init_params(init_param: list) -> list:
    """
    基于初始参数生成 MPL 参数组合（约束 beta = gamma）。

    参数：
        init_param (list): [L0, A, alpha, B]

    返回值：
        list: 每组为 [L0, A, alpha, B, C, beta]
    """
    L0, A, alpha, B = init_param
    init_C_param = [1.0]
    init_beta_param = [0.5]
    return list(product([L0], [A], [alpha], [B], init_C_param, init_beta_param))


def mpl_adam_fit(
    data,
    train_set,
    test_set,
    init_params,
    fig_folder,
    eval_interval=FIT_EVAL_INTERVAL,
    lr1=FIT_LR1,
    lr2=FIT_LR2,
    max_steps=FIT_MAX_STEPS,
    grad_norm_thr=FIT_GRAD_NORM_THR,
    loss_thr=FIT_LOSS_THR,
    patience=FIT_PATIENCE
):
    """
    使用 AdamW 优化器对 MPL 模型进行拟合。

    参数：
        data (dict)：包含步骤、学习率和损失的数据集。
        train_set (list)：训练文件名列表。
        test_set (list)：测试文件名列表。
        init_params (list)：MPL 模型的初始参数组。
        fig_folder (str)：保存损失曲线的文件夹路径。
        eval_interval (int)：记录评估的间隔步数（默认来自 config）。
        lr1 (float)：用于 L0、A、B、C 参数的学习率（默认来自 config）。
        lr2 (float)：用于 alpha、beta、gamma 参数的学习率（默认来自 config）。
        max_steps (int)：最大训练步数（默认来自 config）。
        grad_norm_thr (float)：梯度范数的停止阈值（默认来自 config）。
        loss_thr (float)：损失改进的停止阈值（默认来自 config）。
        patience (int)：在无改进的情况下继续训练的步数（默认来自 config）。

    返回值：
        tuple：最优参数组和对应的最小损失值。
    """
    logger = logging.getLogger(__name__)
    logger.info("开始使用 AdamW 进行 MPL 拟合")

    torch_data = preprocess_data(data, train_set + test_set)
    best_params, best_loss = None, float('inf')

    # 遍历所有初始参数组，进行训练
    for init_param in init_params:
        logger.info(f"Initializing with parameters: {init_param}")
        model = MPL(*init_param)  # unpack 6 params
    
        optimizer = torch.optim.AdamW([
            {"params": [model.L0, model.A, model.B, model.C], "lr": lr1},
            {"params": [model.alpha, model.beta], "lr": lr2},  # gamma 被移除
        ])


        loss_history, min_loss, steps_no_improve = [], float('inf'), 0

        for step in tqdm(range(max_steps), desc="训练进度"):
            total_loss = compute_loss(model, torch_data, train_set, optimizer)
            loss_history.append(total_loss.item())

            if total_loss < min_loss - loss_thr:
                min_loss = total_loss.item()
                steps_no_improve = 0
            else:
                steps_no_improve += 1

            if step > patience and steps_no_improve >= patience:
                logger.info(f"提前停止于 step {step}：{patience} 步无改进。")
                break

            grad_norm = compute_grad_norm(model)
            if grad_norm < grad_norm_thr:
                logger.info(f"提前停止于 step {step}：梯度范数 {grad_norm:.2e} 小于阈值 {grad_norm_thr:.2e}")
                break

            if total_loss < best_loss:
                best_loss = total_loss.item()
                best_params = [p.item() for p in model.parameters()]
                logger.info(f"发现更优损失：{best_loss}")

            if step % eval_interval == 0:
                log_step(step, total_loss, best_loss, model, grad_norm)

        plot_loss_curve(loss_history, fig_folder)

    logger.info(f"拟合完成。最优损失: {best_loss}, 最优参数: {best_params}")
    return best_params, best_loss

with open(r"C:\Users\lenovo\Desktop\processed_data_50_after2500.pkl", "rb") as f:
    data = pickle.load(f)

results = {}

# 每个样本作为自己的训练集和测试集
for name in data:
    train_set = [name]
    test_set = [name]

    init_param = initialize_params(data, train_set)
    init_params = generate_init_params(init_param)

    best_params, best_loss = mpl_adam_fit(
        data=data,
        train_set=train_set,
        test_set=test_set,
        init_params=init_params,
        fig_folder="results",
        eval_interval=FIT_EVAL_INTERVAL,
        lr1=FIT_LR1,
        lr2=FIT_LR2,
        max_steps=1000,
        grad_norm_thr=FIT_GRAD_NORM_THR,
        loss_thr=FIT_LOSS_THR,
        patience=80
    )

    results[name] = {
        "best_params": best_params,
        "best_loss": best_loss
    }

# 打印结果
for k, v in results.items():
    print(f"\n📘 {k}")
    print("✅ 最优参数:", v["best_params"])
    print(f"✅ 最小损失: {v['best_loss']:.6f}")
