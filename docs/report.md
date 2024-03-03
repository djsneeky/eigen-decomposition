## Eigen Image Analysis

### Generating Gaussian random vectors

![alt text](../img/w_scatter_plot.png)

![alt text](../img/x_tilde_scatter_plot.png)

![alt text](../img/x_scatter_plot.png)

### Covariance Estimation and Whitening

#### Theoretical Covariance

```py
RX = [[2, -1.2],
      [-1.2, 1]]
```

#### Estimated Covariance

```py
R_hat = [[ 1.97972664 -1.16673842]
          [-1.16673842  0.94524608]]
```

#### Scatter Plots for X_tilde_i and W_i

![alt text](../img/x_tilde_est_scatter.png)

![alt text](../img/w_est_scatter.png)

#### Covariance estimation R_hat_W

```py
R_hat_whitened = 
    [[ 1.00000000e+00 -1.70700958e-16]
    [-1.70700958e-16  1.00000000e+00]]
 ```

### Eigenimages, PCA, and Data Reduction

![alt text](../img/eigenimages_12.png)

![alt text](../img/proj_coeff.png)

![alt text](../img/synth_a.png)

![alt text](../img/orig_a.png)

### Image Classification

### Misclassification using Rk

| Input Character | Classifier Output |
| --------------- | ----------------- |
| d               | a                 |
| j               | y                 |
| l               | i                 |
| n               | v                 |
| q               | a                 |
| u               | a                 |
| y               | v                 |

### Misclassification of Bk variations

#### Bk = lambda_k

| Input Character | Classifier Output |
| --------------- | ----------------- |
| i               | l                 |
| y               | v                 |

#### Bk = Rwc

| Input Character | Classifier Output |
| --------------- | ----------------- |
| g               | q                 |
| y               | v                 |

#### Bk = Diagonal of Rwc

| Input Character | Classifier Output |
| --------------- | ----------------- |
| f               | t                 |
| y               | v                 |

#### Bk = I

| Input Character | Classifier Output |
| --------------- | ----------------- |
| f               | t                 |
| g               | q                 |
| y               | v                 |

### Tradeoff of data model accuracy vs estimate accuracy

The more complex the covariance model, the better it may fit the training data. But this may also lead to overfitting and poor generalization for unseen data. Simplifying the covariance model can reduce variance, but may introduce bias. The trade-off invovles finding the right balance between model complexity and available data, aiming to achieve a model that captures the data distribution while remaining general to unseen data.
