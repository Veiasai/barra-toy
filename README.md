# Barra Learning Workspace

这个仓库用于学习 Barra 风格多因子模型、最小二乘截面回归、风险分解、组合优化和相关数学基础。

## 快速运行

基础脚本只依赖 `numpy` 和 `pandas`：

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python barra_toy_example.py
```

交互式学习工具使用 `streamlit`、`plotly` 和 `scipy`：

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

在 VS Code SSH Remote 里，把远程端口 `8501` 转发到本地，然后在浏览器打开 Streamlit 给出的本地地址。

## 仓库内容

### 样例脚本

- [barra_toy_example.py](/home/devops/barra/barra_toy_example.py): 一个独立可运行的 Barra 玩具样例，生成模拟股票数据、构造风格/行业因子暴露、模拟股票收益，并通过截面加权回归估计每日因子收益。

### 计算模块

- [barra_lab/data.py](/home/devops/barra/barra_lab/data.py): 生成模拟股票 universe、构造因子暴露、模拟因子收益和个股收益。
- [barra_lab/regression.py](/home/devops/barra/barra_lab/regression.py): 执行截面加权回归，输出因子收益、收益分解、矩阵中间量和诊断指标。
- [barra_lab/risk.py](/home/devops/barra/barra_lab/risk.py): 计算因子协方差矩阵、特质方差、股票协方差矩阵和组合风险分解。
- [barra_lab/optimizer.py](/home/devops/barra/barra_lab/optimizer.py): 用 alpha、协方差矩阵、交易成本和约束做组合优化。
- [barra_lab/plots.py](/home/devops/barra/barra_lab/plots.py): Plotly 图表辅助函数。
- [barra_lab/ui.py](/home/devops/barra/barra_lab/ui.py): Streamlit 页面配置、侧边栏参数和场景缓存辅助函数。

### 交互页面

- [app.py](/home/devops/barra/app.py): 总览页，展示从模拟数据到风险矩阵的完整流水线。
- [pages/1_Factor_Exposures.py](/home/devops/barra/pages/1_Factor_Exposures.py): 因子暴露页面，可调 winsorize、标准化、股票数量和风险参数。
- [pages/2_Regression_Steps.py](/home/devops/barra/pages/2_Regression_Steps.py): 截面回归页面，可直接编辑单日股票收益和回归权重，并展示 `X`、`W`、`X'WX`、伪逆和残差。
- [pages/3_Risk_Model.py](/home/devops/barra/pages/3_Risk_Model.py): 风险模型页面，展示 `F`、`D`、`Sigma`、特征值和组合风险分解。
- [pages/4_Optimizer.py](/home/devops/barra/pages/4_Optimizer.py): 优化器页面，可编辑 alpha、基准权重、当前权重、风险厌恶、交易成本和约束。

### 学习笔记

- [barra_factor_model_notes.md](/home/devops/barra/barra_factor_model_notes.md): Barra 多因子模型复习笔记，覆盖因子暴露、因子收益、协方差矩阵、特质风险、正定性和优化器。
- [optimizer.md](/home/devops/barra/optimizer.md): Barra 优化器笔记，覆盖目标函数、风险惩罚、交易成本、约束、无闭式解算法和收敛证明思路。
- [calculus_optimization_notes.md](/home/devops/barra/calculus_optimization_notes.md): 微积分与优化基础笔记，覆盖最小二乘、矩阵求导、梯度、Hessian、梯度下降、神经网络反向传播和可导性。

## 核心公式

Barra 风格收益分解：

```text
r = X f + epsilon
```

截面加权回归估计因子收益：

```text
f_hat = (X' W X)^(-1) X' W r
```

股票协方差矩阵：

```text
Sigma = X F X' + D
```

组合风险：

```text
Var(portfolio) = w' Sigma w
```

Barra 风险分解形式：

```text
w' Sigma w = (X'w)' F (X'w) + w'Dw
```

组合优化目标的一种常见写法：

```text
max_w alpha'w - lambda/2 * w' Sigma w - transaction_cost(w)
```

## 学习路径

建议按这个顺序看：

1. 运行 [barra_toy_example.py](/home/devops/barra/barra_toy_example.py)，先看完整数据流。
2. 阅读 [barra_factor_model_notes.md](/home/devops/barra/barra_factor_model_notes.md)，理解 Barra 模型主线。
3. 阅读 [calculus_optimization_notes.md](/home/devops/barra/calculus_optimization_notes.md)，补齐最小二乘和梯度基础。
4. 阅读 [optimizer.md](/home/devops/barra/optimizer.md)，理解如何把风险模型放进组合优化器。
5. 再看 `barra_lab/` 下的模块化实现。

## 当前定位

这个仓库是学习和实验用途，不是生产级风险模型。真实 Barra 或机构风险系统还会涉及：

- 更完整的因子定义和数据清洗
- 行业/国家/风格约束
- 因子正交化
- 因子协方差估计与半衰期
- 特质风险模型
- 风险预测校准
- 交易成本模型
- 组合约束和执行系统
