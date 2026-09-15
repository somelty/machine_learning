'''
任务1 初验：用作业给出的手算例子检验 KNN 实现，输出贴报告附录
（必做口径：多数投票 / 近邻均值，weight=False）
'''
import numpy as np
from q1_task1_knn import KNN

# 分类手算例子：A类(1,1),(2,1),(2,3)；B类(4,4),(5,3)；查询 q=(3,2)，k=3 → A ---
x_train = np.array([[1, 1], [2, 1], [2, 3], [4, 4], [5, 3]])
y_train = np.array(["A", "A", "A", "B", "B"])
pred = KNN(k=3, p=2, method="classification").fit(x_train, y_train).predict(np.array([[3, 2]]))
print(f"分类例子 q=(3,2), k=3: 预测 = {pred[0]}（期望 A）")
assert pred[0] == "A"

# 回归手算例子：(1,2.0),(2,2.5),(3,4.0),(5,5.0),(7,9.0)；查询 x=4 ---
x_train = np.array([[1], [2], [3], [5], [7]])
y_train = np.array([2.0, 2.5, 4.0, 5.0, 9.0])
pred2 = KNN(k=2, p=2, method="regression").fit(x_train, y_train).predict(np.array([[4]]))
pred3 = KNN(k=3, p=2, method="regression").fit(x_train, y_train).predict(np.array([[4]]))
print(f"回归例子 x=4, k=2: 预测 = {pred2[0]:.4f}（期望 4.5000）")
print(f"回归例子 x=4, k=3: 预测 = {pred3[0]:.4f}（期望 3.8333）")
assert np.isclose(pred2[0], 4.5)
assert np.isclose(pred3[0], 3.83333333)

print("\n全部例子通过")
