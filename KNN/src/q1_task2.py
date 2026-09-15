import csv
from q1_task1_knn import KNN
import q1_task1_validator
from sklearn.datasets import load_iris
import requests
import pandas as pd
from io import BytesIO
import os
import time

def format_result(result):
    '''
    格式化输出结果到 terminal，便于查看
    :param result: 验证后的结果 dict 类型
    :return: None

    result = {
        "曼哈顿": [cur1, cur2, cur3, ...],
        "欧式": [...]
    }

    cur = {
        # 分类
        "accuracy": {"mean":.., "std":..},
        "time_cost": ..,   # 浮点数，不是子字典！单独加的耗时

        # 回归
        "mse": {"mean": .., "std": ..},
        "mae": {"mean": .., "std": ..},
        "r2": {"mean": .., "std": ..},
        "time_cost": ..
    }
    '''
    for p_label in result.keys():
        print(f"===== 距离：{p_label} =====")
        k_list = list(range(1, 16, 2))
        for idx, cur in enumerate(result[p_label]):
            k = k_list[idx]
            print(f"k = {k}")
            for arg, info in cur.items():
                if arg == "time_cost":
                    print(f"  {arg}: {info:.4f} s")
                else:
                    print(f"  {arg}: {info['mean']:.4f} ± {info['std']:.4f}")
        print()


# ========== 加载数据集 ==========
# 加载 iris
iris = load_iris()
x_iris = iris.data
y_iris = iris.target

# 加载 concrete
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "concrete.csv")
if not os.path.exists(file_path):
    resp = requests.get(url)
    df = pd.read_excel(BytesIO(resp.content), engine="xlrd")
    df.to_csv(file_path, index=False)
else:
    df = pd.read_csv(file_path)

x_concrete = df.iloc[:, :-1].values
y_concrete = df.iloc[:, -1].values


def save_csv(result, name, method, weight):
    '''实验结果展平为长表 CSV 存到 ../report/tables/，便于报告制表'''
    table_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'report', 'tables')
    os.makedirs(table_dir, exist_ok=True)
    fname = f"task2_{name}_{method}_{'weighted' if weight else 'unweighted'}.csv"
    rows = []
    for p_label, cur_list in result.items():
        for k, cur in zip(range(1, 16, 2), cur_list):
            for metric, info in cur.items():
                if metric == "time_cost":
                    rows.append([p_label, k, metric, f"{info:.4f}", ""])
                else: # 分类：accuracy    回归：mse/mae/r^2
                    rows.append([p_label, k, metric, f"{info['mean']:.4f}", f"{info['std']:.4f}"])
    with open(os.path.join(table_dir, fname), 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["distance", "k", "metric", "mean", "std"])
        writer.writerows(rows)
    print(f"结果已保存: report/tables/{fname}")


def experiment(x, y, method: str, weight: bool = False, name: str = "dataset"):
    print(f"任务:{method}，距离加权:{weight}")
    result = {"曼哈顿": [], "欧式": []}
    for p_label, p_val in [("曼哈顿", 1), ("欧式", 2)]:
        for i in range(1, 16, 2):
            t_start = time.perf_counter()
            model = KNN(i, p_val, method, weight)
            cur = q1_task1_validator.k_fold_validate(model, x, y)
            t_end = time.perf_counter()
            cost = t_end - t_start
            cur["time_cost"] = cost
            result[p_label].append(cur)
    format_result(result)
    save_csv(result, name, method, weight)

if __name__ == "__main__":
    # 分类近邻 / 回归近邻均值
    experiment(x_iris, y_iris, "classification", name="iris")
    experiment(x_concrete, y_concrete, "regression", name="concrete")
    # 距离加权投票 / 加权均值
    experiment(x_iris, y_iris, "classification", weight=True, name="iris")
    experiment(x_concrete, y_concrete, "regression", weight=True, name="concrete")