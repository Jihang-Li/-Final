import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize
from itertools import product
from tqdm import tqdm
from matplotlib.font_manager import FontProperties
import pandas as pd

def lr(current_step,
       max_lr=1e-3,
       min_lr=1e-4,
       total_steps=60000,
       warmup_steps=500, 
       lr_method='cosine'):

    if lr_method == 'constant':
        return max_lr

    if current_step <= warmup_steps:
        return max_lr

    num_steps_ = current_step - warmup_steps
    annealing_steps_ = total_steps - warmup_steps
    delta_lr = max_lr - min_lr

    decay_ratio = float(num_steps_) / float(annealing_steps_)

    if lr_method == 'linear':
        coeff = (1.0 - decay_ratio)
        current_lr = min_lr + coeff * delta_lr

    elif lr_method == 'cosine':
        coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        current_lr = min_lr + coeff * delta_lr

    elif lr_method == '811':  # 8-1-1 三阶段：η, η/√10, η/10
        if decay_ratio < 0.8:
            current_lr = max_lr
        elif decay_ratio < 0.9:
            current_lr = max_lr / math.sqrt(10)
        else:
            current_lr = max_lr / 10.0


    elif lr_method == 'wsd':  # 仅最后 20% 做退火
        if decay_ratio < 0.8:
            current_lr = max_lr
        else:
            r = (decay_ratio - 0.8) / 0.2
            coeff = 0.5 * (math.cos(math.pi * r) + 1.0)
            current_lr = min_lr + coeff * delta_lr

    elif lr_method == 'cosine10':  # 最后 10% 退火
        if decay_ratio < 0.9:
            current_lr = max_lr
        else:
            r = (decay_ratio - 0.9) / 0.1
            coeff = 0.5 * (math.cos(math.pi * r) + 1.0)
            current_lr = min_lr + coeff * delta_lr

    elif lr_method == 'cosine20':  # 最后 20% 退火
        if decay_ratio < 0.8:
            current_lr = max_lr
        else:
            r = (decay_ratio - 0.8) / 0.2
            coeff = 0.5 * (math.cos(math.pi * r) + 1.0)
            current_lr = min_lr + coeff * delta_lr

    elif lr_method == 'cyclic':  # 5个周期
        num_cycles = 5
        cycle_pos = (num_steps_ % (annealing_steps_ // num_cycles)) / (annealing_steps_ / num_cycles)
        coeff = 0.5 * (math.cos(math.pi * cycle_pos) + 1.0)
        current_lr = min_lr + coeff * delta_lr

    else:
        raise Exception('{} decay style is not supported.'.format(lr_method))

    return current_lr


def Howe_Scaling_Law(step,lr_method,L0,A,C,alpha):
    predict_loss = L0 + A*(1/S1[lr_method][step])**alpha -C*S2[lr_method][step]
    return predict_loss

def huber_loss(residual,delta):
    return np.where(np.abs(residual)<delta,0.5*(residual)**2, delta*np.abs(residual) - 0.5*(delta**2))

def objective(params):
    loss = 0
    for fitting_lr_method in fitting_lr_methods:
        indices = [i for i, lr_method in enumerate(lr_methods_data) if fitting_lr_method == lr_method]
        predict_losses = Howe_Scaling_Law(fitting_steps[indices], fitting_lr_method, *params)
        residual = fitting_losses[indices] - predict_losses
        loss += np.mean(residual ** 2)
    return loss

## 提取数据，作拟合##
df = pd.read_pickle("C:/Users/lenovo/gpt_loss+lrs.pkl")
scheduler_keys = list(df.keys())

# 结果容器
all_fitting_steps = {}
all_fitting_losses = {}
all_learning_rates = {}

# 遍历每个调度器
for key in scheduler_keys:
    raw = df[key]
    method = key.split("scheduler:")[-1].split("_")[0]
    total_steps = raw['step'].max()
    
    # 每隔1000步选一个点
    select_steps = np.arange(2000, total_steps + 1, 25)
    step_list = []
    loss_list = []
    lr_list = []

    # 提取并记录数据
    for s in select_steps:
        nearest_idx = np.argmin(np.abs(raw['step'] - s))
        step_value = int(raw['step'].iloc[nearest_idx])
        loss_value = float(raw['Metrics/loss'].iloc[nearest_idx])
        lr_value = lr(step_value, lr_method=method, max_lr=2e-4, min_lr=2e-5, total_steps=total_steps)

        step_list.append(step_value)
        loss_list.append(loss_value)
        lr_list.append(lr_value)

    all_fitting_steps[method] = np.array(step_list)
    all_fitting_losses[method] = np.array(loss_list)
    all_learning_rates[method] = np.array(lr_list)


# 初始化参数搜索空间
L0_init_range = np.linspace(0.1, 2.1, 2)
A_init_range = np.linspace(1, 22, 3)
C_init_range = np.linspace(1, 22, 3)
alpha_init_range = np.linspace(0, 0.8, 3)

decay_factor = 0.999
total_steps = 33907

# 提取调度器名字
fitting_lr_methods = list(all_fitting_steps.keys())

# 初始化 S1, S2
lrs = {}
S1 = {}
momentum = {}
S2 = {}

for lr_method in fitting_lr_methods:
    steps = np.arange(0, total_steps + 1)
    lrs[lr_method] = np.array([
        lr(step, lr_method=lr_method, max_lr=1e-3, min_lr=1e-4, total_steps=total_steps,warmup_steps=200)
        for step in steps
    ])
    S1[lr_method] = np.cumsum(lrs[lr_method])
    momentum[lr_method] = np.zeros_like(lrs[lr_method])
    for i in range(1, len(steps)):
        momentum[lr_method][i] = (
            decay_factor * momentum[lr_method][i - 1]
            + (lrs[lr_method][i - 1] - lrs[lr_method][i])
        )
    S2[lr_method] = np.cumsum(momentum[lr_method])

# 定义 scaling law 函数
def Howe_Scaling_Law(step, lr_method, L0, A, C, alpha):
    step = int(step)
    return L0 + A * (1.0 / S1[lr_method][step])**alpha - C * S2[lr_method][step]

# 拟合
optimal_params = {}

for lr_method in fitting_lr_methods:
    print(f"\n>>> 开始拟合：{lr_method}")
    steps = all_fitting_steps[lr_method]
    losses = all_fitting_losses[lr_method]

    def objective(params):
        predicted = [Howe_Scaling_Law(step, lr_method, *params) for step in steps]
        residual = losses - np.array(predicted)
        return np.mean(residual ** 2)

    best_params = None
    best_loss = np.inf
    initial_params = product(L0_init_range, A_init_range, C_init_range, alpha_init_range)
    
    for init in tqdm(initial_params, desc=f"Fitting {lr_method}"):
        result = minimize(objective, init, method='L-BFGS-B',
                          bounds=[(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)],
                          options={'maxiter': 100000, 'ftol': 1e-9, 'gtol': 1e-6, 'eps': 1e-8})
        if result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x

    optimal_params[lr_method] = best_params
    print(f"{lr_method} best (L0, A, C, alpha):", best_params)

##绘图##
#对所有调度器（811, wsd, cosine）进行交叉预测，绘制 3x3 子图（含 R²）

fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True, sharey=True)
axes = axes.flatten()

methods = ["811", "wsd", "cosine"]
steps = np.arange(0, total_steps + 1)

plot_idx = 0
for fit_method in methods:
    L0, A, C, alpha = optimal_params[fit_method]
    for eval_method in methods:
        preds = [Howe_Scaling_Law(step, eval_method, L0, A, C, alpha) for step in steps]
        true_y = all_fitting_losses[eval_method]
        true_x = all_fitting_steps[eval_method]
        pred_at_truth = [Howe_Scaling_Law(step, eval_method, L0, A, C, alpha) for step in true_x]
        ss_res = np.sum((true_y - np.array(pred_at_truth))**2)
        ss_tot = np.sum((true_y - np.mean(true_y))**2)
        r2 = 1 - ss_res / ss_tot

        ax = axes[plot_idx]
        ax.plot(steps, preds, label=f'Predicted by {fit_method}', color=colors[methods.index(fit_method)])
        ax.scatter(true_x, true_y, label=f'{eval_method} Ground Truth', color=colors[methods.index(eval_method)], s=10, marker='x')
        ax.set_title(f'{fit_method.upper()} on {eval_method.upper()}\nR² = {r2:.4f}')
        ax.grid()
        ax.legend(fontsize=8)
        ax.set_xlim(0, 33750)
        ax.set_ylim(2.6, 3.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

        plot_idx += 1

plt.tight_layout()
plt.show()
