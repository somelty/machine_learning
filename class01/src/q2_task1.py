'''
题目二 任务1:三种距离方案的距离构造(欧氏复用题目一 KNN)
- MahalanobisKNN:fit 时在训练数据上估计 Σ 与白化矩阵 Σ^(-1/2),predict 白化后走题目一 KNN
- ReliefFKNN:fit 时在训练数据上学习特征权重 w,predict 按 sqrt(w) 缩放特征后走题目一 KNN
- make_synthetic:合成数据(10 有效维高斯簇 + 30 维均匀噪声,3 类)
两类在 CV 中每折新建实例,变换矩阵/权重只在训练折上学习,测试时只读,无数据泄漏。
'''
import numpy as np
from q1_task1_knn import KNN

class MahalanobisKNN:
    '''马氏距离 KNN:Σ 由训练数据估计,距离 = ||Σ^(-1/2)(x_i - x_j)||'''

    def __init__(self, k: int = 3, method: str = "classification"):
        self.k = k
        self.knn = KNN(k=k, p=2, method=method)  # 白化后马氏距离 = 欧氏距离
        self.whiten = None  # Σ^(-1/2),d×d 对称矩阵
        self.cov = None  # 估计的协方差(供初验核对)

    def fit(self, x_train, y_train):
        x = np.asarray(x_train, dtype=float)
        # 手写无偏协方差估计:Σ = XᶜᵀXᶜ/(n-1)
        xc = x - x.mean(axis=0)
        self.cov = xc.T @ xc / (len(x) - 1)
        # 特征分解求 Σ^(-1/2)(手写求逆:特征值取倒数并开根号)
        vals, vecs = np.linalg.eigh(self.cov)
        vals = np.maximum(vals, 1e-10 * np.maximum(vals.max(), 1.0))  # 微小正则防奇异
        self.whiten = (vecs / np.sqrt(vals)) @ vecs.T
        self.knn.fit(self.transform(x), y_train)
        return self

    def transform(self, x):
        '''白化变换,训练与测试共用同一口径'''
        return np.asarray(x, dtype=float) @ self.whiten.T

    def predict(self, x_test):
        return self.knn.predict(self.transform(x_test))


class ReliefFKNN:
    '''ReliefF 加权距离 KNN:权重 w 由训练数据学习,距离 = sqrt(Σ_r w_r (x_ir - x_jr)²)'''

    def __init__(self, k: int = 3, method: str = "classification",
                 k_relief: int = 5, use_prior: bool = True, m: int = None):
        self.k = k
        self.k_relief = k_relief  # 类内/类外近邻数
        self.use_prior = use_prior  # near-miss 是否按先验 P(c)/(1-P(c_i)) 加权(手算初验用 False)
        self.m = m  # 抽样迭代次数,None=遍历全部训练样本
        self.knn = KNN(k=k, p=2, method=method)  # sqrt(w) 缩放后加权距离 = 欧氏距离
        self.w = None  # 特征权重

    def fit(self, x_train, y_train, sample_order=None):
        x = np.asarray(x_train, dtype=float)
        y = np.asarray(y_train)
        self.w = self._relieff(x, y, sample_order)
        self.knn.fit(self.transform(x), y)
        return self

    def transform(self, x):
        '''按 sqrt(max(w,0)) 缩放特征;负权重的特征被剔除(不参与距离)'''
        return np.asarray(x, dtype=float) * np.sqrt(np.maximum(self.w, 0.0))

    def predict(self, x_test):
        return self.knn.predict(self.transform(x_test))

    def _relieff(self, x, y, sample_order=None):
        '''ReliefF 权重更新:对每个抽样样本,同类找 k_relief 个 near-hit,
        每个异类找 k_relief 个 near-miss(可选先验加权),逐特征平方差累加'''
        n, d = x.shape
        classes, counts = np.unique(y, return_counts=True)
        priors = dict(zip(classes, counts / n))
        m = self.m if self.m is not None else n
        if sample_order is not None:  # 手算初验需要指定抽样顺序
            sample_idx = np.asarray(sample_order)
        elif m < n:
            sample_idx = np.random.RandomState(42).choice(n, size=m, replace=False)
        else:
            sample_idx = np.arange(n)
        denom = m * self.k_relief
        w = np.zeros(d)
        for i in sample_idx:
            diff2 = (x - x[i]) ** 2  # n×d,到样本 i 的逐特征平方差
            total = diff2.sum(axis=1)
            c_i = y[i]
            for c in classes:
                idx_c = np.where((y == c) & (np.arange(n) != i))[0]
                kk = min(self.k_relief, len(idx_c))
                top = idx_c[np.argsort(total[idx_c], kind='stable')[:kk]]
                if c == c_i:  # near-hit:同类近邻,减去贡献
                    w -= diff2[top].sum(axis=0) / denom
                else:  # near-miss:每个异类近邻,加上贡献
                    coeff = priors[c] / (1 - priors[c_i]) if self.use_prior else 1.0
                    w += coeff * diff2[top].sum(axis=0) / denom
        return w


def make_synthetic(n_per_class: int = 200, d_effective: int = 10, d_noise: int = 30, seed: int = 42):
    '''合成数据:3 类各 n_per_class 样本;d_effective 维高斯簇(类中心 0/2/4,σ=1),
    后 d_noise 维 U(0,1) 均匀噪声;返回 (x, y)'''
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    for c in range(3):
        center = c * 2.0
        xs.append(rng.normal(center, 1.0, size=(n_per_class, d_effective)))
        ys.append(np.full(n_per_class, c))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    noise = rng.uniform(0, 1, size=(len(x), d_noise))
    return np.hstack([x, noise]), y
