'''
题目二 任务1 初验:用作业 3.1(马氏)/ 3.2(ReliefF)手算例子检验实现,输出贴报告附录
'''
import numpy as np
from q2_task1 import MahalanobisKNN, ReliefFKNN

# --- 3.1 马氏距离手算例子 ---
# 4 个二维样本:均值 (2, 1.25),Σ = [[2/3, 0], [0, 4.75/3]]
x = np.array([[1., 1.], [2., 3.], [3., 1.], [2., 0.]])
y = np.array([0, 0, 0, 0])
m = MahalanobisKNN().fit(x, y)
print("估计的协方差 Σ:")
print(np.round(m.cov, 4))
assert np.allclose(m.cov, [[2 / 3, 0], [0, 4.75 / 3]], atol=1e-4)

xt = m.transform(x)
d13 = np.linalg.norm(xt[0] - xt[2])
d24 = np.linalg.norm(xt[1] - xt[3])
print(f"马氏距离 d(x1,x3) = {d13:.3f}（期望 2.449），d(x2,x4) = {d24:.3f}（期望 2.384）")
assert np.isclose(d13, 2.449, atol=1e-3) and np.isclose(d24, 2.384, atol=1e-3)

e13 = np.linalg.norm(x[0] - x[2])
e24 = np.linalg.norm(x[1] - x[3])
print(f"欧氏距离 d(x1,x3) = {e13:.3f} < d(x2,x4) = {e24:.3f}，"
      f"马氏给出相反排序（{d13:.3f} > {d24:.3f}）")
assert e13 < e24 and d13 > d24

# --- 3.2 ReliefF 手算例子 ---
# 5 样本 4 特征 2 类;m=2 次迭代(抽样 x1、x4),k=1,无先验加权
x = np.array([[2., 1., 0., 5.],
              [3., 1., 0., 9.],
              [1., 0., 0., 3.],
              [8., 4., 7., 6.],
              [9., 5., 6., 8.]])
y = np.array(["A", "A", "A", "B", "B"])
rf = ReliefFKNN(k=3, k_relief=1, use_prior=False, m=2).fit(x, y, sample_order=[0, 3])
w = rf.w
print(f"ReliefF 平均权重 w = ({w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f}, {w[3]:.1f})"
      f"（期望 29.5, 8, 48.5, 1）")
assert np.allclose(w, [29.5, 8.0, 48.5, 1.0], atol=1e-6)

print("\n全部手算例子通过")
