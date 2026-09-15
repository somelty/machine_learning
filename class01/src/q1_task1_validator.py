import numpy as np


def k_fold_divide(samples_n, fold: int = 5, seed: int = 123):
    '''
    :param samples_n: 样本数据量
    :param fold: 几折交叉验证
    :param seed: 种子
    :return: 每一折的验证集索引与训练集索引
    '''
    rng = np.random.RandomState(seed)
    idx = np.arange(samples_n)
    rng.shuffle(idx)
    size_per_fold = samples_n // fold
    r = samples_n % fold
    size_per_fold_list = np.full(fold, size_per_fold, dtype=int)
    size_per_fold_list[:r] += 1
    fold_idx_list = []
    i = 0
    for size in size_per_fold_list:
        s, e = i, i + size
        val_idx = idx[s:e]
        train_idx = np.concatenate([idx[:s], idx[e:]])
        fold_idx_list.append((val_idx, train_idx))
        i = e
    return fold_idx_list


def k_fold_validate(model, x_train: np.ndarray, y_train: np.ndarray, fold: int = 5, seed: int = 123):
    '''
    k 折交叉验证
    :param model: 模型，如：KNN
    :param x_train: 训练数据集 feature
    :param y_train: 训练数据集 label
    :param fold: 几折交叉验证
    :param seed: 随机种子，便于复现
    :return: 分类任务返回准确率 acc，回归任务返回 MSE, MAE, R^2
    '''
    fold_idx_list = k_fold_divide(len(x_train), fold, seed)
    # backup = (model.x_train, model.y_train, model.mean, model.std, model.fitted)
    acc_list = []
    mse_list = []
    mae_list = []
    r2_list = []
    for fold_idx in fold_idx_list:
        val_idx, train_idx = fold_idx
        x_tr, y_tr = x_train[train_idx], y_train[train_idx]
        x_val, y_val = x_train[val_idx], y_train[val_idx]
        model.fit(x_tr, y_tr)
        y_pred = model.predict(x_val)
        if model.method == "classification":
            acc_list.append(_accuracy(y_val, y_pred))
        elif model.method == "regression":
            mse_list.append(_mse(y_val, y_pred))
            mae_list.append(_mae(y_val, y_pred))
            r2_list.append(_r2(y_val, y_pred))

    # 收集结果
    result = {}
    if acc_list:
        arr = np.array(acc_list)
        result["accuracy"] = {"mean": arr.mean(), "std": arr.std()}
    if mse_list:
        arr = np.array(mse_list)
        result["mse"] = {"mean": arr.mean(), "std": arr.std()}
    if mae_list:
        arr = np.array(mae_list)
        result["mae"] = {"mean": arr.mean(), "std": arr.std()}
    if r2_list:
        arr = np.array(r2_list)
        result["r2"] = {"mean": arr.mean(), "std": arr.std()}
    # model.x_train, model.y_train, model.mean, model.std, model.fitted = backup
    return result


def _accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def _mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def _mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def _r2(y_true, y_pred):
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    if sst == 0:
        return 0.0
    return 1 - sse / sst
