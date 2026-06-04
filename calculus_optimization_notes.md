# 微积分与优化基础笔记

这份笔记整理关于最小二乘、矩阵求导、梯度、Hessian、梯度下降、神经网络反向传播和约束优化的讨论。

## 1. 最小二乘是什么

最小二乘的英文是：

```text
least squares
```

它的目标是：

```text
找一组参数，让预测值和真实值之间的误差平方和最小。
```

设：

```text
y ~= X beta
```

残差：

```text
e = y - X beta
```

目标函数：

```text
S(beta) = e'e = (y - X beta)'(y - X beta)
```

这里 `e'e` 等于所有残差平方和：

```text
e'e = e1^2 + e2^2 + ... + en^2
```

平方的作用：

- 避免正负误差抵消
- 对大误差惩罚更重
- 让目标函数有良好的数学性质

## 2. 最小二乘闭式解

目标函数展开：

```text
S(beta) = (y - X beta)'(y - X beta)
        = y'y - 2 beta'X'y + beta'X'X beta
```

对 `beta` 求梯度：

```text
gradient S(beta) = -2X'y + 2X'X beta
```

令梯度为 0：

```text
-2X'y + 2X'X beta = 0
```

得到正规方程：

```text
X'X beta = X'y
```

如果 `X'X` 可逆：

```text
beta_hat = (X'X)^(-1) X'y
```

这就是普通最小二乘的闭式解。

## 3. 矩阵逆和运算规则

`A^(-1)` 表示 `A` 的逆矩阵。

定义：

```text
A A^(-1) = A^(-1) A = I
```

矩阵乘法有结合律：

```text
(AB)C = A(BC)
```

矩阵乘法一般没有交换律：

```text
AB != BA
```

逆矩阵乘积规则：

```text
(AB)^(-1) = B^(-1) A^(-1)
```

证明思路是直接验证：

```text
(AB)(B^(-1)A^(-1)) = I
```

以及：

```text
(B^(-1)A^(-1))(AB) = I
```

## 4. 偏导、梯度和 Hessian

偏导是多变量函数对某一个变量的导数，求导时把其他变量当常数。

如果：

```text
f(x, y) = x^2 + 3xy + y^2
```

则：

```text
partial f / partial x = 2x + 3y
partial f / partial y = 3x + 2y
```

梯度是所有偏导组成的列向量：

```text
gradient f =
[partial f / partial x
 partial f / partial y]
```

Hessian 是所有二阶偏导组成的矩阵：

```text
H_ij = partial^2 f / partial xi partial xj
```

在最小二乘里：

```text
gradient S(beta) = -2X'y + 2X'X beta
```

对梯度再求导：

```text
H = 2X'X
```

## 5. 为什么 `X'X` 是对称矩阵

转置规则：

```text
(AB)' = B'A'
```

所以：

```text
(X'X)' = X'(X')' = X'X
```

因此 `X'X` 是对称矩阵。

从元素角度看，`X'X` 的第 `i,j` 个元素是 `X` 第 `i` 列和第 `j` 列的内积。

内积满足：

```text
x_i' x_j = x_j' x_i
```

所以矩阵关于主对角线对称。

## 6. 矩阵求导的两个常用结果

令：

```text
c = X'y
A = X'X
```

线性项：

```text
beta'c = beta1 c1 + beta2 c2 + ... + betak ck
```

所以：

```text
gradient(beta'c) = c
```

因此：

```text
gradient(-2 beta'X'y) = -2X'y
```

二次型：

```text
beta'A beta
```

一般有：

```text
gradient(beta'A beta) = (A + A') beta
```

如果 `A` 对称：

```text
gradient(beta'A beta) = 2A beta
```

在最小二乘中 `A = X'X`，所以：

```text
gradient(beta'X'X beta) = 2X'X beta
```

## 7. 矩阵接近奇异为什么会让回归不稳定

最小二乘闭式解是：

```text
beta_hat = (X'X)^(-1) X'y
```

矩阵逆可以理解成矩阵版的除法。

如果 `X'X` 在某个方向上非常小，接近 0，那么取逆以后，这个方向会变得非常大。

一维类比：

```text
1 / 0.001 = 1000
```

所以如果数据里有一点点噪声：

```text
X'y = 真实值 + 小噪声
```

乘上很大的逆矩阵后，小噪声会被放大成很大的系数变化。

直观例子：

```text
Value 因子
Cheapness 因子
```

如果这两个因子几乎一样，回归就很难判断收益应该归给谁。

今天可能估出：

```text
Value 收益 = +2%
Cheapness 收益 = -1.5%
```

明天稍微换一点数据，可能变成：

```text
Value 收益 = -1%
Cheapness 收益 = +1.5%
```

但两者合起来对股票收益的解释可能差不多。

所以问题不是模型完全不能拟合，而是单个系数或单个因子收益拆分变得不稳定。

表现为：

- 系数很大
- 正负乱跳
- 样本稍微变化，结果大变
- 因子解释不可靠
- 优化器可能过度反应

一句话：

```text
(X'X)^(-1) 很大，就像除以接近 0 的数，会把数据里的小噪声放大成很大的系数变化。
```

## 8. 梯度的几何意义

单变量函数的导数是切线斜率。

多变量函数有无数方向可以走。梯度表示在某一点函数增长最快的方向。

在某个点 `x` 附近：

```text
f(x + d) ~= f(x) + gradient f(x)' d
```

其中：

```text
gradient f(x)' d
```

是沿方向 `d` 的一阶变化量。

如果规定：

```text
||d|| = 1
```

则根据内积公式：

```text
gradient f(x)' d = ||gradient f(x)|| ||d|| cos(theta)
```

当 `d` 与梯度同方向时，`cos(theta)=1`，变化率最大。

所以：

```text
梯度方向 = 上升最快方向
负梯度方向 = 下降最快方向
```

## 9. 内积是什么

内积，也叫 dot product。

如果：

```text
a = [a1, a2, ..., an]'
b = [b1, b2, ..., bn]'
```

则：

```text
a'b = a1b1 + a2b2 + ... + anbn
```

几何公式：

```text
a'b = ||a|| ||b|| cos(theta)
```

因此：

- 内积为正：大致同方向
- 内积为 0：互相垂直
- 内积为负：大致反方向

## 10. 梯度下降法

梯度下降法用于最小化函数：

```text
min f(x)
```

更新公式：

```text
x_new = x_old - eta * gradient f(x_old)
```

其中：

- `eta` 是步长，也叫 learning rate
- `gradient f(x_old)` 是当前点上升最快方向
- 负梯度方向是下降最快方向

步长太小会慢，步长太大可能震荡或发散。

## 11. 非凸函数和梯度下降

如果函数是凸的，梯度下降在合适条件下可以收敛到全局最优。

如果函数是非凸的，梯度下降通常不能保证找到全局最优，也不能保证解唯一。

非凸问题可能有：

- 多个局部最小值
- 鞍点
- 平坦区域
- 震荡区域

神经网络就是典型非凸优化问题。

实践中，训练神经网络通常追求：

```text
验证集效果好，而不是数学上证明找到了全局最优。
```

## 12. 神经网络里的梯度

神经网络不是没有解析表达式。

它通常是很多简单函数的复合：

```text
y_hat = W2 * ReLU(W1 x + b1) + b2
```

损失函数例如：

```text
L = mean((y - y_hat)^2)
```

这些操作都有明确的数学表达式。

神经网络梯度通常不是数值差分估计，而是通过自动微分和反向传播计算。

反向传播本质是链式法则：

```text
dL/dw = dL/da * da/dz * dz/dw
```

标准神经网络层大多可导或几乎处处可导：

- Linear
- Conv
- ReLU
- sigmoid
- tanh
- softmax
- normalization
- pooling

ReLU 在 0 点不可导，但实践中使用次梯度处理。

## 13. 带约束的梯度优化

普通梯度下降可能走出可行域。

例如组合权重要求：

```text
sum(w) = 1
w_i >= 0
```

普通更新后可能不满足这些约束。

常见处理方式：

投影梯度法：

```text
z = w_k - eta * gradient f(w_k)
w_{k+1} = Projection_C(z)
```

其中 `C` 是可行域。

变量变换：

```text
w = softmax(z)
```

这样天然满足：

```text
w_i >= 0
sum(w) = 1
```

惩罚函数法：

```text
new_objective = 原目标 + penalty * constraint_violation
```

拉格朗日和 KKT 方法用于更系统地处理约束。

在组合优化里，如果问题是凸的，通常优先使用 QP 或凸优化求解器。

## 14. 连续和可导

可导一定连续：

```text
可导 => 连续
```

所以：

```text
不连续 => 不可导
```

但连续不一定可导。

例如：

```text
f(x) = |x|
```

它在 `x=0` 连续，但不可导，因为左导数和右导数不同。

## 15. 一句话总结

最小二乘、Barra 回归、组合优化和神经网络训练，本质都离不开同一套思想：

```text
定义目标函数，
理解它的导数结构，
用梯度或更高级的优化算法寻找让目标函数变小的参数。
```
