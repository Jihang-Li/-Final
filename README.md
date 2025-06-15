# 📐 Scaling Law with Learning Rate Annealing & MPL Fitting

本项目包括两部分内容：

1. **Scaling Law with Learning Rate Annealing (LRA)** 的损失预测与交叉评估；
2. **Multi-Power Law (MPL)** 模型损失预测与交叉评估。
3. **Multi-Power Law (MPL)** 模型改进版本的损失预测与交叉评估。
---

## ✅ 使用说明

### 1. Scaling Law with LRA 拟合

- 模型训练步数的范围可在主文件中第 **114 行** 调节：

  ```python
  for step in range(start_step, end_step):  # 第114行附近
  ```

- 支持不同类型的学习率调度策略（如 `cosine`、`linear`、`wsd`、`811` 等），可在对应函数中指定。

---

### 2. MPL 模型拟合与交叉验证预测

- MPL 拟合与预测基于 `.pkl` 格式的预处理文件，例如：

  ```
  processed_data_50_after2500.pkl
  ```

  加载方式如下：

  ```python
  import pickle
  with open("processed_data_50_after2500.pkl", "rb") as f:
      data = pickle.load(f)
  ```

- 支持 `β = γ` 和 `β ≠ γ` 两种参数版本；
- 多模型参数与数据文件的交叉预测以表格 + 九宫格图像输出，图像自动保存在指定文件夹；
- 输出包括常见指标：MSE、R²、MAE、MAPE、MaxPE、RMSE；
- 推荐使用 `evaluate_mpl_pair_grid_style()` 或其 `beta=gamma` 特化版本。

---

### 3. 配置文件说明（config）

- 为确保参数调用顺利，`MPL` 拟合代码依赖于原作者提供的 `src/config.py` 配置文件；



- 项目结构如下：


  project_root/
  ├── processed_data_50_after2500.pkl
  ├── src/
  │   ├── models.py
  │   ├── utils.py
  │   ├── config.py     
  │   └── fitting.py
  ├── results_crossfit/
  │   └── grid_fit.png
  ├── your_driver_script.py
  └── README.md
  ```

