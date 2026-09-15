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




## 报告生成

双击 `report/build_html.bat`：渲染 HTML 并在浏览器打开（编辑时看效果）；双击 `report/build_pdf.bat`：生成 PDF（运行前先关闭已打开的 report_full.pdf）。手动等价命令：

```bash
cd report
pandoc report_full.md -s -o report_full.html --mathml --css=style.css --toc
# 用 Edge 无头模式把 HTML 打印成 PDF（路径按本机安装位置调整）
"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless --disable-gpu \
  --print-to-pdf="report_full.pdf" --no-pdf-header-footer --virtual-time-budget=15000 \
  "file:///D:/code/ML/class01/report/report_full.html"
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
- `report/report_full.md`：整体报告源文件（题目一+题目二合并版，唯一编辑入口，姓名学号需自行填写）
- `report/report_q1.md`、`report_q2.md`：分题源文件（已合并进 report_full.md，无需再改）
- `report/report_full.pdf`：实验报告
