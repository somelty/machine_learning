import numpy as np

class KNN:
    '''
    使用 sklearn 规范完成 KNN 的功能
    '''

    def __init__(self, k: int = 3, p: int = 2, method: str = "classification", weight: bool = False, normalize: bool = False, median: bool = False):
        '''
        模型的基本初始化
        :param k: 近邻范围
        :param p: 距离参数
        :param method: KNN 完成任务类型
        :param weight: 是否对距离加权
        :param normalize: 是否开启特征标准化，True开启，False关闭
        :param median: 回归时是否用近邻中位数预测（True 时忽略 weight）
        '''
        self.y_train: np.ndarray | None = None
        self.x_train: np.ndarray | None = None
        self.k = k
        self.p = p
        self.method = method
        self.weight = weight
        self.normalize = normalize
        self.median = median
        self.fitted = False
        # 标准化需要保存的统计量
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "KNN":
        '''
        训练模型（传入data）
        :param x_train: 数据的 feature
        :param y_train: 数据的 label
        :return: 返回自身实例
        '''
        self.x_train = np.asarray(x_train)
        if self.x_train.ndim == 1:
            self.x_train = self.x_train.reshape(-1, 1)  # 单特征数据按列向量处理
        self.y_train = np.asarray(y_train)

        if self.normalize:
            self.mean = np.mean(self.x_train, axis=0)
            self.std = np.std(self.x_train, axis=0)
            self.std[self.std < 1e-8] = 1.0  # 防止除0 除以1变为零
        else:
            self.mean = self.std = None  # 清掉旧统计量，避免之后重开 normalize 时用错

        self.fitted = True
        return self

    def predict(self, x_test: np.ndarray):
        '''
        进行预测
        :param x_test: 待预测的数据
        :return: 返回类型为 ndarray，含义为：预测结果
        '''
        if not self.fitted:
            raise ValueError("还未训练")
        if self.k >= len(self.x_train):
            raise ValueError('k 过大')

        x_test = np.asarray(x_test)
        if x_test.ndim == 1:
            # 1D 输入有歧义：长度等于特征数视为单个样本，否则视为多个单特征样本
            if x_test.shape[0] == self.x_train.shape[1]:
                x_test = x_test.reshape(1, -1)
            else:
                x_test = x_test.reshape(-1, 1)
        if x_test.shape[1] != self.x_train.shape[1]:
            raise ValueError(f"特征维度不匹配：x_test 为 {x_test.shape[1]} 维，训练数据为 {self.x_train.shape[1]} 维")
        x_train = self.x_train
        if self.normalize:
            x_test = (x_test - self.mean) / self.std
            x_train = (x_train - self.mean) / self.std

        dist = self._distance(x_test, x_train)
        sorted_idx = np.argsort(dist, axis=1, kind="stable")[:, :self.k]  # 稳定排序：等距邻居按原始顺序，结果确定
        if self.method == "classification":
            return self._classification(sorted_idx, dist)
        elif self.method == "regression":
            return self._regression(sorted_idx, dist)
        else:
            raise ValueError("method 应为 'classification' 或 'regression'")

    def _distance(self, x_test: np.ndarray, x_train: np.ndarray) -> np.ndarray:
        '''
        使用闵式公式计算距离
        :param x_test: 待预测的数据
        :param x_train: 训练数据（是否标准化由调用方处理）
        :return: 返回类型二维数组，含义为：测试数据 1、2 ... 分别到所有 x_train 的距离
        '''
        diff = np.abs(x_test[:, np.newaxis, :] - x_train[np.newaxis, :, :])
        dist = np.power(np.sum(np.power(diff, self.p), axis=2), 1 / self.p)
        return dist

    def _classification(self, sorted_idx: np.ndarray, dist: np.ndarray):
        '''
        分类任务
        :param sorted_idx: 提取的前 k 个 dist 里距离最近的 x_train 的数组下标
        :param dist: 测试数据 1、2 ... 到 x_train 的距离（加权投票时需要）
        :return: 返回类型一维数组，含义为：测试数据 1、2 ... 的预测类别
        '''
        k_labels = self.y_train[sorted_idx]
        predict = []
        if not self.weight:
            # 多数投票
            for one_row in k_labels:
                dict_count = {}
                for label in one_row:
                    dict_count[label] = dict_count.get(label, 0) + 1
                # 多数投票；平票时取距离最近者的类别
                pre = max(dict_count, key=dict_count.get)
                predict.append(pre)
        else:
            # 距离加权投票：票重 w = 1/(d+ε)
            k_dist = np.take_along_axis(dist, sorted_idx, axis=1)
            weights = 1 / (1e-10 + k_dist)
            for labels, w_row in zip(k_labels, weights):
                dict_weight = {}
                for label, w in zip(labels, w_row):
                    dict_weight[label] = dict_weight.get(label, 0.0) + w
                pre = max(dict_weight, key=dict_weight.get)
                predict.append(pre)
        return np.asarray(predict)

    def _regression(self, sorted_idx: np.ndarray, dist: np.ndarray):
        '''
        回归任务
        :param sorted_idx: 提取的前 k 个 dist 里距离最近的 x_train 的数组下标
        :param dist: 测试数据 1、2 ... 到 x_train 的距离
        :return: 返回类型一维数组，含义为：测试数据 1、2 ... 的回归结果
        '''
        k_nums = self.y_train[sorted_idx]
        if self.median:
            return np.median(k_nums, axis=1)  # 中位数预测，没加权的
        if not self.weight:
            return np.mean(k_nums, axis=1)
        k_dist = np.take_along_axis(dist, sorted_idx, axis=1)
        weights = 1 / (1e-10 + k_dist)
        return np.sum(weights * k_nums, axis=1) / np.sum(weights, axis=1)