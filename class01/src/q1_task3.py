'''
任务3
Q1：k 对偏差/方差的影响
Q2：标准化前后对比
Q3：欧氏、曼哈顿（一致性 + 离群点的验证）
Q4：近邻均值、中位数
'''
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from q1_task1_knn import KNN
import q1_task1_validator as v

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, '..', 'report', 'figures')
TABLE_DIR = os.path.join(BASE_DIR, '..', 'report', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

K_VALS = [1, 3, 5, 7, 9, 11, 13, 15]
DISTANCES = [('曼哈顿', 1), ('欧氏', 2)]
CV_SEED = 123 # 控制 cv 划分，保证训练集与验证集划分一样，使得 mse 的差异来自离群点


def load_data():
    '''
    加载数据集
    :return: iris 的 features, label
             concrete 的 features, label
    '''
    iris = load_iris()
    df = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'concrete.csv'))
    return iris.data, iris.target, df.iloc[:, :-1].values, df.iloc[:, -1].values


def cv_scores(x, y, k, p, method, weight=False, normalize=False, median=False):
    '''做一次 5 折 CV'''
    model = KNN(k, p, method, weight=weight, normalize=normalize, median=median)
    return v.k_fold_validate(model, x, y)


def save_rows_csv(rows, fname, header):
    '''
    保存做实验过程种产生的 csv 文件
    :param rows: 文件内容
    :param fname: 文件名
    :param header: 文件头
    :return: None
    '''
    with open(os.path.join(TABLE_DIR, fname), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"表格已保存: report/tables/{fname}")


def _run_curve(x, y, k, p, method, **kw):
    ''' 对一个 k 跑 一次 CV，返回 (指标名, mean, std)
    这里的指标选取的是：
        对于分类任务，选取 accuracy 的 mean 来评估模型的最优 k
        对于回归任务，选取 mse 的 mean 来评估模型的最优 k
    '''
    r = cv_scores(x, y, k, p, method, **kw)
    metric = 'accuracy' if method == 'classification' else 'mse'
    return metric, r[metric]['mean'], r[metric]['std']


# ============ Q1 ============
def q1_k_curve(x_iris, y_iris, x_conc, y_conc):
    '''
    第一小问，k 对偏差/方差的影响
    :param x_iris: iris 特征集
    :param y_iris: iris 标签
    :param x_conc: concrete 特征集
    :param y_conc: concrete 标签
    :return:
    '''
    iris_means, iris_stds, conc_means, conc_stds = [], [], [], []
    # 对每一个 k 进行一次5折 cv检验，使用四个 list 收集每个 k 的 cv 结果里的 mean 和 std，具体 cv 选取指标在 _run_curve 函数注释有写
    for k in K_VALS:
        _, im, istd = _run_curve(x_iris, y_iris, k, 2, 'classification')
        _, cm, cstd = _run_curve(x_conc, y_conc, k, 2, 'regression')
        iris_means.append(im); iris_stds.append(istd)
        conc_means.append(cm); conc_stds.append(cstd)
    # 转为 ndarray 便于后续操作
    iris_means, iris_stds = np.array(iris_means), np.array(iris_stds)
    conc_means, conc_stds = np.array(conc_means), np.array(conc_stds)
    # 选取使得各自指标最好的对应 k 值的 idx
    best_idx_iris = int(np.argmax(iris_means))
    best_idx_conc = int(np.argmin(conc_means))

    # 画图来表示 k 与对应选取指标的关系
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].errorbar(K_VALS, iris_means, yerr=iris_stds, marker='o', capsize=5)
    axes[0].set_xticks(K_VALS)  # 刻度对齐实际 k 值,避免自动刻度落在偶数位
    axes[0].axvline(K_VALS[best_idx_iris], color='red', linestyle='--',
                    label=f'最优 k={K_VALS[best_idx_iris]}')
    axes[0].set_xlabel('k'); axes[0].set_ylabel('准确率')
    axes[0].set_title('Iris：准确率-k 曲线（欧氏，无加权）')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].errorbar(K_VALS, conc_means, yerr=conc_stds, marker='o', capsize=5)
    axes[1].set_xticks(K_VALS)
    axes[1].axvline(K_VALS[best_idx_conc], color='red', linestyle='--',
                    label=f'最优 k={K_VALS[best_idx_conc]}')
    axes[1].set_xlabel('k'); axes[1].set_ylabel('MSE')
    axes[1].set_title('Concrete：MSE-k 曲线（欧氏，无加权）')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    # 保存图
    fig.savefig(os.path.join(FIG_DIR, 'q1_k_effect.png'), dpi=150)
    plt.close(fig)
    # 保存表格
    save_rows_csv([[k, f"{m:.4f}", f"{s:.4f}", f"{cm:.4f}", f"{cs:.4f}"]
                   for k, m, s, cm, cs in zip(K_VALS, iris_means, iris_stds, conc_means, conc_stds)],
                  'q1_k_curves.csv',
                  ['k', 'iris_accuracy_mean', 'iris_accuracy_std', 'concrete_mse_mean', 'concrete_mse_std'])

    print("\nQ1")
    print(f"Iris 最优 k={K_VALS[best_idx_iris]}，准确率={iris_means[best_idx_iris]:.4f}±{iris_stds[best_idx_iris]:.4f}")
    print(f"Concrete 最优 k={K_VALS[best_idx_conc]}，MSE={conc_means[best_idx_conc]:.4f}±{conc_stds[best_idx_conc]:.4f}")
    return iris_means, iris_stds, conc_means, conc_stds


# ============ Q2：标准化前后对比 ============
def q2_normalize(x_iris, y_iris, x_conc, y_conc):
    rows = []
    curves = {}
    # 对不同的数据集与任务，进行不同距离测量方式，而每个距离下跑标准化与不标准化版的，对标准化与不标准化再运行不同的 k
    for name, x, y, method in [('iris', x_iris, y_iris, 'classification'),
                               ('concrete', x_conc, y_conc, 'regression')]:
        metric_name = 'accuracy' if method == 'classification' else 'mse'
        for p_label, p in DISTANCES:
            for norm in (False, True):
                means, stds = [], []
                for k in K_VALS:
                    _, m, s = _run_curve(x, y, k, p, method, normalize=norm)
                    means.append(m); stds.append(s)
                    rows.append([name, p_label, '标准化' if norm else '未标准化', k,
                                 metric_name, f"{m:.4f}", f"{s:.4f}"])
                curves.setdefault((name, p_label), {})['标准化' if norm else '未标准化'] = (means, stds)

    # 将实验数据保存为 q2_normalize.csv
    save_rows_csv(rows, 'q2_normalize.csv',
                  ['dataset', 'distance', 'normalize', 'k', 'metric', 'mean', 'std'])

    # 四张子图：行=数据集(Concrete/Iris)，列=距离(曼哈顿/欧氏)，各画标准化前后图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, p_label, title, ylabel) in zip(
            axes.ravel(),
            [('concrete', '曼哈顿', 'Concrete：标准化前后 MSE-k 曲线（曼哈顿）', 'MSE'),
             ('concrete', '欧氏', 'Concrete：标准化前后 MSE-k 曲线（欧氏）', 'MSE'),
             ('iris', '曼哈顿', 'Iris：标准化前后准确率-k 曲线（曼哈顿）', '准确率'),
             ('iris', '欧氏', 'Iris：标准化前后准确率-k 曲线（欧氏）', '准确率')]):
        for label, (means, stds) in curves[(name, p_label)].items():
            ax.errorbar(K_VALS, means, yerr=stds, marker='o', capsize=5, label=label)
        ax.set_xticks(K_VALS)
        ax.set_xlabel('k'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q2_normalize.png'), dpi=150)
    plt.close(fig)
    print("图已保存: report/figures/q2_normalize.png")

    # 最优 k 处标准化前后对比
    for name, x, y, method in [('concrete', x_conc, y_conc, 'regression'),
                               ('iris', x_iris, y_iris, 'classification')]:
        metric_name = 'accuracy' if method == 'classification' else 'mse'
        for norm in (False, True):
            best = None
            for k in K_VALS:
                _, m, s = _run_curve(x, y, k, 2, method, normalize=norm)
                if best is None or (m < best[0] if method == 'regression' else m > best[0]):
                    best = (m, s, k)
            print(f"[Q2] {name} {'标准化' if norm else '未标准化'}(欧氏): 最优 k={best[2]}, {metric_name}={best[0]:.4f}±{best[1]:.4f}")


# ============ Q3：欧氏 vs 曼哈顿（距离对比 + 离群点注入实验） ============
def q3_distance_compare(x_iris, y_iris, x_conc, y_conc):
    '''第三问前半：两种距离在两个数据集上的 k 曲线对比'''
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, x, y, method, title, ylabel) in zip(
            axes,
            [('iris', x_iris, y_iris, 'classification', 'Iris：两种距离的准确率-k 曲线', '准确率'),
             ('concrete', x_conc, y_conc, 'regression', 'Concrete：两种距离的 MSE-k 曲线', 'MSE')]):
        for p_label, p in DISTANCES:
            means, stds = [], []
            for k in K_VALS:
                _, m, s = _run_curve(x, y, k, p, method)
                means.append(m); stds.append(s)
                rows.append([name, p_label, k, f"{m:.4f}", f"{s:.4f}"])
            ax.errorbar(K_VALS, means, yerr=stds, marker='o', capsize=5, label=p_label)
        ax.set_xticks(K_VALS)
        ax.set_xlabel('k'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q3_distance_compare.png'), dpi=150)
    plt.close(fig)
    print("图已保存: report/figures/q3_distance_compare.png")
    save_rows_csv(rows, 'q3_distance_compare.csv', ['dataset', 'distance', 'k', 'mean', 'std'])


def _cv_mse_with_injection(x, y, k, p, rate=0.0, seed=0):
    '''
    随机选取一些样本的特征，放大 20 倍，模拟离群点
    :param x: features
    :param y: labels
    :param k: 几折
    :param p: 距离
    :param rate: 离群点的样本比例
    :param seed: 随机种子
    :return: 注入离群点进行 5 折交叉验证后的 MSE
    '''
    rng = np.random.RandomState(seed)
    mses = []
    for val_idx, train_idx in v.k_fold_divide(len(y), 5, CV_SEED):
        # 复制验证集的数据
        x_val = x[val_idx].copy()
        if rate > 0:
            n_inject = max(1, int(round(rate * len(val_idx)))) # 最少也要有 1 个样本
            rows = rng.choice(len(val_idx), size=n_inject, replace=False) # 选中的样本
            cols = rng.randint(0, x.shape[1], size=n_inject) # 选中的特征
            x_val[rows, cols] *= 20  # 单个特征放大 20 倍
        model = KNN(k, p, 'regression').fit(x[train_idx], y[train_idx])
        # MSE
        mses.append(np.mean((y[val_idx] - model.predict(x_val)) ** 2))
    return float(np.mean(mses))


def q3_injection(x_conc, y_conc):
    k = 5  # 固定 k，控制变量
    rates = [0.02, 0.05] # 两次离群点注入，更能看到趋势
    seeds = [1, 2, 3, 4, 5] # 这里使用了5个seed做5次随机抽样，每个seed都做两次离群点注入，减少抽样带来的误差
    baseline = {p_label: _cv_mse_with_injection(x_conc, y_conc, k, p) for p_label, p in DISTANCES} # 未注入离群点的基准
    increases = {p_label: {rate: [] for rate in rates} for p_label, _ in DISTANCES}
    rows = []
    for p_label, p in DISTANCES:
        for rate in rates:
            for seed in seeds:
                injected = _cv_mse_with_injection(x_conc, y_conc, k, p, rate=rate, seed=seed)
                increases[p_label][rate].append(injected - baseline[p_label])
        for rate in rates:
            arr = np.array(increases[p_label][rate])
            rows.append([p_label, f"{rate:.0%}", f"{np.mean(arr):.2f}", f"{np.std(arr, ddof=1):.2f}"])
            print(f"[Q3] {p_label} 注入 {rate:.0%}: MSE 增幅 = {np.mean(arr):.2f} ± {np.std(arr, ddof=1):.2f}（基线 {baseline[p_label]:.2f}）")
    save_rows_csv(rows, 'q3_injection.csv', ['distance', 'rate', 'mse_increase_mean', 'mse_increase_std'])

    # 画图表示
    fig, ax = plt.subplots(figsize=(7, 4))
    xpos = np.arange(len(rates))
    width = 0.35
    for i, (p_label, _) in enumerate(DISTANCES):
        means = [np.mean(increases[p_label][r]) for r in rates]
        stds = [np.std(increases[p_label][r], ddof=1) for r in rates]
        ax.bar(xpos + i * width, means, width, yerr=stds, capsize=5, label=p_label)
    ax.set_xticks(xpos + width / 2)
    ax.set_xticklabels([f'注入 {r:.0%}' for r in rates])
    ax.set_ylabel('MSE 增幅')
    ax.set_title(f'Concrete 离群点注入下的鲁棒性对比（k={k}，5 次随机重复）')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q3_injection.png'), dpi=150)
    plt.close(fig)
    print("图已保存: report/figures/q3_injection.png")


# ============ Q4：近邻均值 vs 中位数 ============
def q4_median(x_conc, y_conc):
    rows = []
    curves = {}
    for p_label, p in DISTANCES:
        for agg, kw in [('均值', {}), ('中位数', {'median': True})]:
            means, stds = [], []
            for k in K_VALS:
                r = cv_scores(x_conc, y_conc, k, p, 'regression', **kw)
                means.append(r['mse']['mean']); stds.append(r['mse']['std'])
                rows.append(['concrete', p_label, agg, k, f"{r['mse']['mean']:.4f}", f"{r['mse']['std']:.4f}"])
            curves.setdefault(p_label, {})[agg] = (means, stds)

    save_rows_csv(rows, 'q4_median.csv', ['dataset', 'distance', 'aggregation', 'k', 'mse_mean', 'mse_std'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (p_label, _) in zip(axes, DISTANCES):
        for agg, (means, stds) in curves[p_label].items():
            ax.errorbar(K_VALS, means, yerr=stds, marker='o', capsize=5, label=f'近邻{agg}')
        ax.set_xticks(K_VALS)
        ax.set_xlabel('k'); ax.set_ylabel('MSE')
        ax.set_title(f'Concrete：均值 vs 中位数（{p_label}，无加权）')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q4_median.png'), dpi=150)
    plt.close(fig)
    print("图已保存: report/figures/q4_median.png")

    for p_label, _ in DISTANCES:
        for agg in ('均值', '中位数'):
            means, _ = curves[p_label][agg]
            bi = int(np.argmin(means))
            print(f"[Q4] {p_label} 近邻{agg}: 最优 k={K_VALS[bi]}, MSE={means[bi]:.4f}")


def main():
    x_iris, y_iris, x_conc, y_conc = load_data()
    print("=" * 20 + " Q1 " + "=" * 20)
    q1_k_curve(x_iris, y_iris, x_conc, y_conc)
    print("=" * 20 + " Q2 " + "=" * 20)
    q2_normalize(x_iris, y_iris, x_conc, y_conc)
    print("=" * 20 + " Q3 " + "=" * 20)
    q3_distance_compare(x_iris, y_iris, x_conc, y_conc)
    q3_injection(x_conc, y_conc)
    print("=" * 20 + " Q4 " + "=" * 20)
    q4_median(x_conc, y_conc)


if __name__ == "__main__":
    main()
