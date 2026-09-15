# 说明


## 环境

- Python 3.10+
- 依赖见 `requirements.txt`

## 复现步骤

```bash
# 先创建虚拟环境并激活，这里略
pip install -r requirements.txt
cd src
python q1_task1_manual_check.py   # 题目一 任务1初验：手算例子复现
python q1_task2.py                # 题目一 任务2：Iris分类 + Concrete回归（结果在 report/tables/）
python q1_task3.py                # 题目一 任务3：图与表分别存在 report/figures、report/tables/
python q2_task1_manual_check.py   # 题目二 任务1：马氏/ReliefF 手算例子复现
python q2_task2.py                # 题目二 任务2实验：3数据集×3方法 CV + 噪声鲁棒性 + 效率（结果存 report/tables/）
python q2_task3.py                # 题目二 任务3五问分析 + 可视化（图存 report/figures/）
```


## 文件说明

- `q1_task1_knn.py`：题目一任务1 KNN复现（内容包括分类/回归、欧氏/曼哈顿距离、距离加权选做、标准化、回归中位数）
- `q1_task1_validator.py`：题目一任务1 交叉验证复现，包括了评估指标（accuracy/MSE/MAE/R²）
- `q1_task1_manual_check.py`：题目一的手算例子验证
- `q1_task2.py`：题目一任务2 实验部分
- `q1_task3.py`：题目一任务3 
- `q2_task1.py`：题目二任务1 三种距离方案（MahalanobisKNN / ReliefFKNN / 欧式距离，这里复用了题目一 KNN）
- `q2_task1_manual_check.py`：题目二 手算例子验证
- `q2_task2.py`：题目二任务2 实验
- `q2_task3.py`：题目二任务3 
- `data/concrete.csv`：混凝土数据集
- `report/KNN-report.md`：整体报告
