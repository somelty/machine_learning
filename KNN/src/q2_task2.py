'''
题目二 任务2:受控实验(一键复现:python q2_task2.py)
- 3 数据集(Wine/WDBC/合成 40 维)× 3 方法(欧氏/马氏/ReliefF)× 5 个 k × 5 折 CV
- 噪声鲁棒性:合成数据 10 维纯净 vs 40 维含噪的准确率对比
- 效率:训练/测试时间随 n(150~600)与 d(10/20/40)的变化
所有 CV 对训练折做 z-score 标准化(统计量只来自训练折,无泄漏);结果存 report/tables/
'''
import csv
import os
import time
import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from q1_task1_knn import KNN
import q1_task1_validator as v
from q2_task1 import MahalanobisKNN, ReliefFKNN, make_synthetic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLE_DIR = os.path.join(BASE_DIR, '..', 'report', 'tables')
os.makedirs(TABLE_DIR, exist_ok=True)

K_VALS = [1, 3, 5, 7, 11]
METHODS = ['欧氏', '马氏', 'ReliefF']


def make_model(method, k):
    if method == '欧氏':
        return KNN(k=k, p=2, method='classification')
    if method == '马氏':
        return MahalanobisKNN(k=k)
    return ReliefFKNN(k=k)


def cv_accuracy(method, x, y, k, seed=123):
    '''5 折 CV:训练折 z-score 标准化(统计量只来自训练折),返回 (acc_mean, acc_std)'''
    accs = []
    for val_idx, train_idx in v.k_fold_divide(len(y), 5, seed):
        x_tr, y_tr = x[train_idx], y[train_idx]
        mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0)
        sd[sd < 1e-8] = 1.0
        model = make_model(method, k).fit((x_tr - mu) / sd, y_tr)
        x_val = (x[val_idx] - mu) / sd
        accs.append(np.mean(y[val_idx] == model.predict(x_val)))
    return float(np.mean(accs)), float(np.std(accs))


def save_rows_csv(rows, fname, header):
    with open(os.path.join(TABLE_DIR, fname), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"表格已保存: report/tables/{fname}")


def load_datasets():
    wine = load_wine()
    wdbc = load_breast_cancer()
    x_syn, y_syn = make_synthetic()  # 40 维含噪
    return {'wine': (wine.data, wine.target, list(wine.feature_names)),
            'wdbc': (wdbc.data, wdbc.target, list(wdbc.feature_names)),
            'synthetic': (x_syn, y_syn, None)}


def experiment_cv():
    '''实验 1:主 CV 矩阵(含噪声鲁棒性所需的纯净/含噪合成数据)'''
    data = load_datasets()
    x_clean, y_clean = make_synthetic(d_noise=0)  # 10 维纯净版
    data['synthetic_clean'] = (x_clean, y_clean, None)
    rows = []
    for name, (x, y, _) in data.items():
        for method in METHODS:
            for k in K_VALS:
                acc, std = cv_accuracy(method, x, y, k)
                rows.append([name, method, k, f"{acc:.4f}", f"{std:.4f}"])
                print(f"{name:16s} {method:6s} k={k:2d}: {acc:.4f} ± {std:.4f}")
        print()
    save_rows_csv(rows, 'q2_task2_cv.csv', ['dataset', 'method', 'k', 'acc_mean', 'acc_std'])


def experiment_efficiency():
    '''实验 2:训练/测试时间随 n、d 变化(各重复 3 次取中位数)'''
    k = 5
    rows = []

    def time_one(method, x_tr, y_tr, x_te):
        model = make_model(method, k)
        trs, tes = [], []
        for _ in range(3):
            t0 = time.perf_counter()
            model.fit(x_tr, y_tr)
            trs.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            model.predict(x_te)
            tes.append(time.perf_counter() - t0)
        return float(np.median(trs)), float(np.median(tes))

    # n 变化:d=40 固定,类别均衡子采样
    x_all, y_all = make_synthetic(n_per_class=400)  # 先生成足够大的池再子采样
    for n_per_class in (50, 100, 150, 200, 300, 400):
        x_sub, y_sub = [], []
        for c in range(3):
            idx = np.where(y_all == c)[0][:n_per_class]
            x_sub.append(x_all[idx]); y_sub.append(y_all[idx])
        x_sub, y_sub = np.concatenate(x_sub), np.concatenate(y_sub)
        n = len(x_sub)
        cut = int(n * 0.8)
        for method in METHODS:
            tr, te = time_one(method, x_sub[:cut], y_sub[:cut], x_sub[cut:])
            rows.append(['n', n, method, f"{tr:.5f}", f"{te:.5f}"])
            print(f"n={n:4d} {method:6s}: 训练 {tr:.5f}s, 测试 {te:.5f}s")
    # d 变化:n=600 固定,取前 10/20/40 维
    x_all, y_all = make_synthetic(n_per_class=200)
    cut = int(len(x_all) * 0.8)
    for d in (10, 20, 40):
        x_d = x_all[:, :d]
        for method in METHODS:
            tr, te = time_one(method, x_d[:cut], y_all[:cut], x_d[cut:])
            rows.append(['d', d, method, f"{tr:.5f}", f"{te:.5f}"])
            print(f"d={d:2d} {method:6s}: 训练 {tr:.5f}s, 测试 {te:.5f}s")
    save_rows_csv(rows, 'q2_task2_efficiency.csv', ['vary', 'value', 'method', 'train_s', 'test_s'])


if __name__ == "__main__":
    print("=" * 20 + " 实验1:主 CV 矩阵 " + "=" * 20)
    experiment_cv()
    print("=" * 20 + " 实验2:效率 " + "=" * 20)
    experiment_efficiency()
