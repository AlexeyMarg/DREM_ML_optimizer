# Dynamic Regressor Extension and Mixing (DREM) Based Optimizer for Machine Learning Problems

This repository presents the application of the dynamic regressor extension and mixing (DREM) procedure to optimization problems in machine learning. The aim of this study is to improve the convergence rate of learning algorithms, a critical aspect for solving problems that require high computational efficiency and performance.

### Overview

The proposed solution is an adaptation of the DREM method for machine learning tasks, specifically for data that is not time-dependent. In this modification, the optimization problem is transformed such that each weight of the model is trained independently through a scalar regression problem. This approach significantly increases the convergence rate compared to traditional methods.

The impact of the proposed optimizer on the convergence of the loss function is analyzed in comparison with stochastic gradient descent (SGD) and Adam.

### Repository Contents

- **`perceptron.ipynb`**: A Jupyter notebook demonstrating the application of DREM to a perceptron for regression and binary classification tasks. The notebook includes a comparative analysis of DREM, stochastic gradient descent (SGD), and Adam optimizers.

## DREM Theory

### Original Problem Statement

The original formulation of the problem is presented as follows [1]. Consider a linear regression model:
$$
y(t) = \mathbf{m}^T(t) \mathbf{w}, \tag{1}
$$

where $y(t) \in \mathbb{R} $ and $\mathbf{m}(t) \in \mathbb{R}^q $ are known, bounded time-dependent functions, and $ \mathbf{w} $ is the vector of unknown parameters to be estimated. It is known that the classical gradient descent method can solve this problem if the vector $\mathbf{m} $ satisfies the persistent excitation condition [2]. To overcome this limitation and improve the convergence rate of the estimation, the authors proposed the following procedure.

### Basic algorithm for time-dependent data

#### **Step 1. Dynamic extension**
Apply $q-1$ distinct linear, dynamically stable operators $H_i$ to regression (1) to obtain $q-1$ additional linear regressions:

$$
y_{f_i}(t) = \mathbf{m}_{f_i}^T(t) \mathbf{w},
$$

where $ y_{f_i}(t) = H_i[y(t)] $ and $ \mathbf{m}_{f_i}^T(t) = H_i[\mathbf{m}^T(t)] $. The operators $ H_i $ can be chosen, for instance, as first-order linear filters:

$$
H_i[\cdot](t) = \frac{\alpha_i}{p + \alpha_i}[\cdot](t), \quad \alpha_i > 0, \quad p = \frac{d}{dt},
$$

or delay operators:

$$
H_i[\cdot](t) = [\cdot](t - d_i), \quad d_i > 0.
$$

#### **Step 2. Construction of the Extended System**

An extended system is constructed as follows:

$$
\mathbf{Y}_e(t) = \mathbf{M}(t) \mathbf{w}, \tag{2}
$$

where:
$$
\mathbf{Y}_e^T(t) = \begin{bmatrix}
y(t) & y_{f_1}(t) & \dots & y_{f_{q-1}}(t)
\end{bmatrix},
\quad
\mathbf{M}^T(t) = \begin{bmatrix}
\mathbf{m}(t) & \mathbf{m}_{f_1}(t) & \dots & \mathbf{m}_{f_{q-1}}(t)
\end{bmatrix}.
$$

Multiply Equation (2) by the adjugate matrix $ \text{adj}\{\mathbf{M}(t)\} $ to obtain $ q $ independent scalar linear regressions:

#### **Step 3. Mixing**
$$
Y_i(t) = \Delta(t) w_i, \quad i = 1, \dots, q, \tag{3}
$$

where $ Y_i(t) $ is the $ i $-th element of the vector $ \mathbf{Y}(t) = \text{adj}\{\mathbf{M}(t)\} \mathbf{Y}_e(t) $, and $ \Delta(t) = \det\{\mathbf{M}(t)\} $.

#### **Step 4. Gradient Descent for Scalar Linear Regressions**

Each unknown parameter can now be estimated using gradient descent for scalar linear regression. The solution to Equation (3) in continuous form is given by [1]:

$$
\dot{\hat{w}}_i(t) = \gamma_i \Delta(t) \big(Y_i(t) - \Delta(t) \hat{w}_i(t)\big), \quad i = 1, \dots, q,
$$

where $ \hat{w}_i(t) $ is the estimate of the parameter $ w_i $.

#### Benefits of DREM

The application of the dynamic regressor extension and mixing (DREM) allows transforming the multi-parameter optimization problem into a set of scalar regression tasks. Notably, since only one direction is optimized at a time, the algorithm typically converges to the optimal solution more quickly.

---

### Application of DREM for NOT time-dependent data

We investigate the applicability of the dynamic regressor extension and mixing (DREM) approach to the problem of training on data that is NOT time dependent. In the considered problem, the use of dynamic operators is not feasible, as the features and target variable are not functions of time. Instead, we adopt an alternative approach for regressor extension:


#### **1. Initialization**
The procedure is performed for each training epoch $ k $, where $ k = 1, 2, \dots, K $.

#### **2. Steps for Each Epoch**

2.1. **Mini-batch creation**  
Randomly split the dataset into mini-batches of size equal to the number of weights. Let the mini-batch data be denoted as:
$$
\mathbf{X}^{b_j} \in \mathbb{R}^{(p+1) \times (p+1)}, \quad \mathbf{Y}_e^j \in \mathbb{R}^{p+1},
$$
where $ j $ is the index of the mini-batch.

2.2. **Mini-batch extension and mixing**  
For each mini-batch, the following equation holds:
$$
\mathbf{Y}_e^j = \mathbf{X}^{b_j} \mathbf{w}, \tag{4}
$$
where $ \mathbf{w} $ is the weight vector. Multiply Equation (4) by the adjugate matrix $ \mathbf{X}^{*j} = \text{adj}\{\mathbf{X}^{b_j}\} $:
$
\mathbf{Y}^j = \text{diag}\{\Delta^j\} \mathbf{w},
$
where:
$
\mathbf{Y}^j = \mathbf{X}^{*j} \mathbf{Y}_e^j, \quad \mathbf{X}^{*j} \mathbf{X}^{b_j} = \begin{bmatrix}
\Delta & 0 & \cdots & 0 \\
0 & \ddots & \cdots & 0 \\
0 & 0 & \cdots & \Delta
\end{bmatrix}, \quad \Delta^j = \det(\mathbf{X}^{b_j}).
$

To reduce computational complexity, instead of explicitly computing the adjugate matrix, we use Cramer's rule:
$
Y_i^j = \det(\mathbf{X}^{(b_j, i)}),
$
where $ \mathbf{X}^{(b_j, i)} $ is the matrix $ \mathbf{X}^{b_j} $ with the $ i $-th column replaced by the vector $ \mathbf{Y}_e^j $ [9].

After the transformations, the multi-parameter regression problem with $ p+1 $ unknowns reduces to solving $ p+1 $ scalar regression problems:
$$
Y_i^j = \Delta^j w_i, \quad i = 1, \dots, p+1.
$$

2.3 **Weight Update**

To determine the weights, use gradient descent:
$$
w_i := w_i - \alpha \Delta^j (\Delta^j w_i - Y_i^j). \tag{5}
$$

To improve stability, the following algorithm can be used instead:
$
w_i := w_i - \alpha \Delta^j \frac{\Delta^j w_i - Y_i^j}{1 + \alpha (\Delta^j)^2}.
$

#### **3. Termination Criterion**

The training process terminates when either the specified number of epochs $ K $ is reached or the change in the loss function becomes smaller than a predefined threshold.

---

### Advantages of the Proposed Approach

This solution enables independent estimation of each weight, significantly increasing the convergence rate and allowing the method to handle datasets with a large number of examples. However, the proposed algorithm may become unstable if the matrix $ \mathbf{X}^{b_j} $ is singular, i.e., when the data exhibits multicollinearity. To address this issue, several techniques can be applied:
- Eliminate feature collinearity during data preprocessing.
- Skip weight updates for mini-batches where the determinant $ \Delta^j $ is close to zero.



### Reference

1. Aranovskiy S., Bobtsov A., Ortega R., Pyrkin A. *Performance enhancement of parameter estimators via dynamic regressor extension and mixing*. IEEE Transactions on Automatic Control, 2016. DOI: [10.1109/TAC.2016.2614889](https://doi.org/10.1109/TAC.2016.2614889)

2. Ljung L., System Identification: Theory for the User. Upper Saddle River, NJ: Prentice-Hall, 1987.

