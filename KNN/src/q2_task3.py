'''
题目二 任务3:比较与分析(一键复现:python q2_task3.py)
Q1:三种方法在各数据集的提升幅度与一致性
Q2:Wine 相关系数矩阵(满矩阵 vs 对角权重讨论)
Q3:合成数据噪声鲁棒性 + ReliefF 噪声特征权重统计
Q4:计算开销随 n、d 变化(实测 + 复杂度)
Q5:ReliefF Wine 权重柱状图、马氏白化 PCA vs 欧氏 PCA 投影
图输出 report/figures/,关键数字打印供报告引用
'''
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
import q1_task1_validator as v
from q2_task1 import MahalanobisKNN, ReliefFKNN, make_synthetic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, '..', 'report', 'figures')
TABLE_DIR = os.path.join(BASE_DIR, '..', 'report', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

K_VALS = [1, 3, 5, 7, 11]
METHODS = ['欧氏', '马氏', 'ReliefF']


def load_cv():
    return pd.read_csv(os.path.join(TABLE_DIR, 'q2_task2_cv.csv'))


def best_acc(df, dataset, method):
    sub = df[(df.dataset == dataset) & (df.method == method)]
    row = sub.loc[sub.acc_mean.idxmax()]
    return row.k, row.acc_mean, row.acc_std


# ============ Q1:提升幅度与一致性 ============
def q1_summary(df):
    print("[Q1] 各数据集最优 k 准确率与相对欧氏的提升:")
    for ds in ['wine', 'wdbc', 'synthetic']:
        base_k, base_acc, base_std = best_acc(df, ds, '欧氏')
        line = f"  {ds:10s} 欧氏 k={base_k} {base_acc:.4f}"
        for m in ['马氏', 'ReliefF']:
            k, acc, std = best_acc(df, ds, m)
            line += f" | {m} k={k} {acc:.4f} (Δ{acc - base_acc:+.4f})"
        print(line)
    # k-敏感性图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, ds, title in zip(axes, ['wine', 'wdbc', 'synthetic'],
                             ['Wine', 'WDBC', '合成数据(40 维)']):
        for m in METHODS:
            sub = df[(df.dataset == ds) & (df.method == m)]
            ax.errorbar(sub.k, sub.acc_mean, yerr=sub.acc_std, marker='o', capsize=4, label=m)
        ax.set_xticks(K_VALS)  # 刻度对齐实际 k 值
        ax.set_xlabel('k'); ax.set_title(f'{title}:准确率-k 曲线')
        ax.legend(); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('准确率')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q2_k_sensitivity.png'), dpi=150)
    plt.close(fig)
    print("图已保存: report/figures/q2_k_sensitivity.png")


# ============ Q2:Wine 相关系数矩阵 ============
def q2_wine_corr():
    wine = load_wine()
    x = (wine.data - wine.data.mean(0)) / wine.data.std(0)
    corr = np.corrcoef(x.T)
    names = wine.feature_names
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((abs(corr[i, j]), names[i], names[j], corr[i, j]))
    print("[Q2] Wine 相关系数最高的 5 对特征:")
    for r, a, b, raw in sorted(pairs, reverse=True)[:5]:
        print(f"  {a} vs {b}: r = {raw:+.3f}")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title('Wine 特征相关系数矩阵(标准化后)')
    fig.colorbar(im)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q2_wine_corr.png'), dpi=150)
    plt.close(fig)
    print("图已保存: report/figures/q2_wine_corr.png")


# ============ Q3:噪声鲁棒性 + ReliefF 权重统计 ============
def fold_weights(x, y, seed=123):
    '''5 折各自在训练折(标准化)上学习 ReliefF 权重,返回 (fold 数 × 特征数) 数组'''
    ws = []
    for val_idx, train_idx in v.k_fold_divide(len(y), 5, seed):
        x_tr, y_tr = x[train_idx], y[train_idx]
        mu, sd = x_tr.mean(0), x_tr.std(0)
        sd[sd < 1e-8] = 1.0
        rf = ReliefFKNN(k=5).fit((x_tr - mu) / sd, y_tr)
        ws.append(rf.w)
    return np.array(ws)


def q3_noise(df):
    # 准确率下降:各方法用纯净数据上的最优 k
    print("[Q3] 合成数据噪声鲁棒性(纯净 10 维 → 含噪 40 维,各方法取纯净数据最优 k):")
    for m in METHODS:
        k, acc_clean, _ = best_acc(df, 'synthetic_clean', m)
        sub = df[(df.dataset == 'synthetic') & (df.method == m) & (df.k == k)]
        acc_noisy, std = sub.acc_mean.iloc[0], sub.acc_std.iloc[0]
        print(f"  {m:6s} k={k}: 纯净 {acc_clean:.4f} → 含噪 {acc_noisy:.4f},降幅 {acc_clean - acc_noisy:+.4f}")
    # 柱状图
    fig, ax = plt.subplots(figsize=(7, 4))
    xpos = np.arange(len(METHODS))
    width = 0.35
    for i, m in enumerate(METHODS):
        k, acc_clean, _ = best_acc(df, 'synthetic_clean', m)
        sub = df[(df.dataset == 'synthetic') & (df.method == m) & (df.k == k)]
        acc_noisy = sub.acc_mean.iloc[0]
        ax.bar(xpos[i] - width / 2, acc_clean, width, label='纯净 10 维' if i == 0 else None)
        ax.bar(xpos[i] + width / 2, acc_noisy, width, label='含噪 40 维' if i == 0 else None)
        ax.text(xpos[i], max(acc_clean, acc_noisy) + 0.005, f'{acc_noisy - acc_clean:+.3f}',
                ha='center', fontsize=10)
    ax.set_xticks(xpos); ax.set_xticklabels(METHODS)
    ax.set_ylabel('准确率'); ax.set_ylim(0.4, 1.08)
    ax.set_title('合成数据注入 30 维噪声特征前后的准确率(柱顶为变化量)')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q2_noise_robustness.png'), dpi=150)
    plt.close(fig)
    print("图已保存: report/figures/q2_noise_robustness.png")

    # ReliefF 学到的权重:有效维 vs 噪声维
    x_noisy, y_noisy = make_synthetic()
    ws = fold_weights(x_noisy, y_noisy)
    w_mean, w_std = ws.mean(0), ws.std(0, ddof=1)
    eff, noi = w_mean[:10], w_mean[10:]
    print(f"[Q3] ReliefF 权重统计(5 折均值±标准差):")
    print(f"  10 个有效维权重均值: {eff.mean():.4f} ± {eff.std():.4f}")
    print(f"  30 个噪声维权重均值: {noi.mean():.4f} ± {noi.std():.4f}")
    print(f"  噪声维中权重 ≤0 的比例: {(noi <= 0).mean():.0%},"
          f" 权重小于有效维均值 5% 的比例: {(noi < 0.05 * eff.mean()).mean():.0%}")
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#d62728'] * 10 + ['#999999'] * 30
    ax.bar(range(40), w_mean, yerr=w_std, color=colors, capsize=3)
    ax.axvline(9.5, color='black', linestyle='--', linewidth=1)
    ax.text(4.5, ax.get_ylim()[1] * 0.95, '10 个有效维', ha='center', fontsize=10)
    ax.text(24.5, ax.get_ylim()[1] * 0.95, '30 个噪声维', ha='center', fontsize=10)
    ax.set_xlabel('特征序号'); ax.set_ylabel('ReliefF 权重(5 折均值)')
    ax.set_title('合成数据上 ReliefF 学到的特征权重:噪声维权重接近 0')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q3_relieff_weights_synthetic.png'), dpi=150)
    plt.close(fig)
    print("图已保存: report/figures/q3_relieff_weights_synthetic.png")


# ============ Q4:效率 ============
def q4_efficiency():
    df = pd.read_csv(os.path.join(TABLE_DIR, 'q2_task2_efficiency.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, vary, title, xlabel in [(axes[0], 'n', '计算时间随样本数 n 的变化(d=40)', '样本数 n'),
                                    (axes[1], 'd', '计算时间随特征数 d 的变化(n=600)', '特征数 d')]:
        sub = df[df.vary == vary]
        for m in METHODS:
            s = sub[sub.method == m]
            ax.plot(s.value, s.train_s, marker='o', label=f'{m} 训练')
            ax.plot(s.value, s.test_s, marker='s', linestyle='--', label=f'{m} 测试')
        ax.set_xlabel(xlabel); ax.set_ylabel('时间(s)')
        ax.set_title(title); ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_xticks([150, 300, 450, 600, 900, 1200])  # 刻度对齐实测点
    axes[1].set_xticks([10, 20, 40])
    axes[0].tick_params(axis='x', labelsize=8); axes[1].tick_params(axis='x', labelsize=8)
    axes[0].set_xscale('log'); axes[1].set_xscale('log')
    axes[0].set_yscale('log'); axes[1].set_yscale('log')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q4_efficiency.png'), dpi=150)
    plt.close(fig)
    print("图已保存: report/figures/q4_efficiency.png")
    # 增长倍数实测
    sub = df[(df.vary == 'n') & (df.method == 'ReliefF')]
    t150 = sub[sub.value == 150].train_s.iloc[0]
    t1200 = sub[sub.value == 1200].train_s.iloc[0]
    print(f"[Q4] ReliefF 训练时间 n=150→1200(8 倍): {t150:.4f}s → {t1200:.4f}s({t1200 / t150:.1f} 倍,≈n²)")
    sub = df[(df.vary == 'n') & (df.method == '马氏')]
    t150 = sub[sub.value == 150].train_s.iloc[0]
    t1200 = sub[sub.value == 1200].train_s.iloc[0]
    print(f"[Q4] 马氏训练时间 n=150→1200: {t150:.4f}s → {t1200:.4f}s(基本不随 n 变化,求逆主导且 d 小)")


# ============ Q5:ReliefF Wine 权重 + 白化 PCA ============
def q5_visual():
    wine = load_wine()
    x, y, names = wine.data, wine.target, wine.feature_names
    xz = (x - x.mean(0)) / x.std(0)
    # ReliefF 权重柱状图(5 折均值)
    ws = fold_weights(x, y)
    w_mean, w_std = ws.mean(0), ws.std(0, ddof=1)
    order = np.argsort(w_mean)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(13), w_mean[order], yerr=w_std[order], capsize=3)
    ax.set_xticks(range(13))
    ax.set_xticklabels([names[i] for i in order], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('ReliefF 权重(5 折均值)')
    ax.set_title('Wine 上 ReliefF 学到的 13 个特征权重')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q5_relieff_weights_wine.png'), dpi=150)
    plt.close(fig)
    print("图已保存: report/figures/q5_relieff_weights_wine.png")
    print("[Q5] Wine 权重排序:", ' > '.join(names[i] for i in order[:5]), '(前5)')

    # 白化 PCA vs 欧氏 PCA 投影
    m = MahalanobisKNN().fit(xz, y)
    x_white = m.transform(xz)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in [(axes[0], xz, '欧氏距离视角:标准化后 PCA'),
                            (axes[1], x_white, '马氏距离视角:$\Sigma^{-1/2}$ 白化后 PCA')]:
        # 手写 PCA:中心化后 SVD
        u, s, vt = np.linalg.svd(data - data.mean(0), full_matrices=False)
        proj = u[:, :2] * s[:2]
        for c, cmap in zip(range(3), ['#1f77b4', '#ff7f0e', '#2ca02c']):
            ax.scatter(proj[y == c, 0], proj[y == c, 1], s=12, label=f'类 {c}')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q5_pca_whiten.png'), dpi=150)
    plt.close(fig)
    print("图已保存: report/figures/q5_pca_whiten.png")


def main():
    df = load_cv()
    q1_summary(df)
    q2_wine_corr()
    q3_noise(df)
    q4_efficiency()
    q5_visual()
    print("\n全部完成:图在 report/figures 下")


if __name__ == "__main__":
    main()
