#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
from abc import ABCMeta, abstractmethod
from .functions import _Function, _function_map
import numpy as np
import pandas as pd
from scipy.stats import norm


__all__ = ['make_pd_function', 'make_ts_function', 'ts_corr', 'ts_cov', 'ts_decay', 'ts_std', 'ts_quantile',
           'ts_mean', 'pn_winsorize', 'pn_normal', 'pn_quantile', 'pn_log', 'pn_inv', 'pn_div', 'pn_z_score',
           'pn_add', 'pn_sub', 'pn_mul', 'pn_sqrt', 'pn_abs', 'pn_neg', 'pn_max', 'pn_min', 'pn_sin', 'pn_cos',
           'pn_tan']


# class _PD_Function(_Function, metaclass=ABCMeta):
#     def __init__(self, function, name, arity, **kwargs):
#         super(_Function, self).__init__(
#             function=function,
#             name=name,
#             arity=arity
#         )
#
#     def __call__(self, *args):
#         return self.function(*args)
class _TS_Function(_Function, metaclass=ABCMeta):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity, **kwargs):
        super(_TS_Function, self).__init__(
            function=function,
            name=name,
            arity=arity
        )
        self.is_ts = kwargs.get('is_ts', False)
        #only for ts
        self.days = kwargs.get('days', [])

    def __call__(self, *args):
        # return self.function(*args)
        if not hasattr(self, 'is_ts') or not self.is_ts:
            return self.function(*args)
        else:
            return self.function(*args, self.win)

    # 新增重设参数d的方法
    def set_win(self, win):
        self.win = win
        self.name += '_%d' % self.win


def _convert_multi_index(arguments):
    def make_wrapper(func):
        def wrapper(*args, **kwargs):
            # 拿到被装饰函数的参数名列表
            code = func.__code__
            names = list(code.co_varnames[:code.co_argcount])
            # args类型是tuple, tuple是不可变对象
            args_in_list = list(args)
            multi_flag = True
            last_series_multi_index = 0
            # 装饰arguments
            for argument in arguments:
                num = names.index(argument)
                value = args[num]
                if isinstance(value, list):
                    for i in range(len(value)):
                        if isinstance(value[i], pd.Series) and isinstance(value[i].index, pd.MultiIndex):
                            value[i] = value[i].unstack()
                if isinstance(value, pd.Series) and isinstance(value.index, pd.MultiIndex):
                    last_series_multi_index = value.index
                    value = value.unstack()
                    # multi_flag = True    # 输入为multi-index时，是否将算子输出也变为multi-index
                args_in_list[num] = value
            new_args = tuple(args_in_list)

            if not multi_flag:
                return func(*new_args, **kwargs)
            else:
                # result = result.reindex(args_in_list[0].index)
                result = func(*new_args, **kwargs).stack()
                if isinstance(last_series_multi_index, pd.MultiIndex):
                    result = result.reindex(last_series_multi_index)
                return result

        return wrapper

    return make_wrapper


#TODO
def make_pd_function(function, name, arity, wrap=False, **kwargs):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    pd_flag: bool , optional (default=True)
        it only change construct args if True, it return Series, else return array

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    # if not isinstance(function, np.ufunc):
    #     if function.__code__.co_argcount != arity:
    #         raise ValueError('arity %d does not match required number of '
    #                          'function arguments of %d.'
    #                          % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # Check output shape
    args = [pd.Series(np.arange(0.0, 10.0, 1), index=_construct_stock_index()) for _ in range(arity)]

    try:
        function(*args, **kwargs)
    except ValueError:
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args, **kwargs), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args, **kwargs).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [pd.Series(np.zeros(10), index=_construct_stock_index()) for _ in range(arity)]

    if not np.all(np.isfinite(function(*args, **kwargs))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [pd.Series(-1 * np.ones(10), index=_construct_stock_index()) for _ in range(arity)]

    if not np.all(np.isfinite(function(*args, **kwargs))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    return _Function(function=function,
                     name=name,
                     arity=arity)

# TODO
def make_ts_function(function, name, arity, days, is_ts=True, **kwargs):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    pd_flag: bool , optional (default=True)
        it only change construct args if True, it return Series, else return array

    """
    #TODO check day list and donot contains 0
    # if not isinstance(arity, int):
    #     raise ValueError('arity must be an int, got %s' % type(arity))
    # # if not isinstance(function, np.ufunc):
    # #     if function.__code__.co_argcount != arity:
    # #         raise ValueError('arity %d does not match required number of '
    # #                          'function arguments of %d.'
    # #                          % (arity, function.__code__.co_argcount))
    # if not isinstance(name, str):
    #     raise ValueError('name must be a string, got %s' % type(name))
    # if not isinstance(wrap, bool):
    #     raise ValueError('wrap must be an bool, got %s' % type(wrap))
    #
    # # Check output shape
    # if pd_flag:
    #     args = [pd.Series(np.arange(0.0, 10.0, 1), index=_construct_stock_index()) for _ in range(arity)]
    # else:
    #     args = [np.ones(10) for _ in range(arity)]
    #
    # try:
    #     function(*args, **kwargs)
    # except ValueError:
    #     raise ValueError('supplied function %s does not support arity of %d.'
    #                      % (name, arity))
    # if not hasattr(function(*args), 'shape'):
    #     raise ValueError('supplied function %s does not return a numpy array.'
    #                      % name)
    # if function(*args).shape != (10,):
    #     raise ValueError('supplied function %s does not return same shape as '
    #                      'input vectors.' % name)
    #
    # # Check closure for zero & negative input arguments
    # if pd_flag:
    #     args = [pd.Series(np.zeros(10), index=_construct_stock_index()) for _ in range(arity)]
    # else:
    #     args = [np.zeros(10) for _ in range(arity)]
    #
    # if not np.all(np.isfinite(function(*args))):
    #     raise ValueError('supplied function %s does not have closure against '
    #                      'zeros in argument vectors.' % name)
    # if pd_flag:
    #     args = [pd.Series(-1 * np.ones(10), index=_construct_stock_index()) for _ in range(arity)]
    # else:
    #     args = [-1 * np.ones(10) for _ in range(arity)]
    # if not np.all(np.isfinite(function(*args))):
    #     raise ValueError('supplied function %s does not have closure against '
    #                      'negatives in argument vectors.' % name)

    return _TS_Function(function=function,
                        name=name,
                        arity=arity, is_ts=is_ts, days=days)


def _construct_stock_index():
    arrays = [['20210901', '20210901', '20210901', '20210901', '20210902', '20210902', '20210902', '20210902',
               '20210903', '20210903'], \
              ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ',
               '000001.SZ', '000002.SZ']]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=['trade_time', 'code'])
    return index


def _pn_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return pd.Series(np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.), index=x1.index)


def _pn_inv(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return pd.Series(np.where(np.abs(x1) > 0.001, 1. / x1, 0.), index=x1.index)


def _pn_div(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return pd.Series(np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.), index=x1.index)


def __pn_quantile(x1, axis=1):
    """
    计算某个值在所在截面上的分位数， by axis。计算方法：（排名-0.5）/非缺失值总数
    :param x1: data to be calculated
    :type x1: pd.Series(MultiIndex) or pd.Dataframe
    :param axis: 0 or 1
    :type axis: int
    :return: quantiles of data
    :rtype: pd.DataFrame
    """
    data_pivot = x1.rank(axis=axis) - 0.5
    data_pivot = data_pivot.div((~np.isnan(data_pivot)).sum(axis=axis), axis=1 - axis)
    return data_pivot


def __pn_cut(x0, bound_mode, lb, ub, cut=False, fill=np.nan, axis=1):
    """
    根据数据在所在截面上的 分位数（或偏离均值的标准差倍数（z-score）） 划出上下界，对超出上下界的数据进行截尾处理（cut）
    bound_mode为'quantile'时，按分位数划分，lb、ub代表下界、上界的分位数。
    bound_mode为'std'时，按z-score划分，lb、ub代表下界、上界的离均值的标准差倍数。

    :param x: data to be cut
    :type x: pd.Series(MultiIndex) or pd.Dataframe
    :param bound_mode: 'quantile' or 'std' : 根据 分位数 或 偏离均值的标准差倍数 划出上下界
    :type bound_mode: str
    :param lb: 下界。小于下界的值会被截除。
    :type lb: float
    :param ub: 上界。大于上界的值会被截除。
    :type ub: float
    :param fill: 上下界外的值被截除后，填充时使用的值。只填充因处于上下界之外而被截除的值，不填充数据中原有缺失值。默认为np.nan，即转为缺失值。
    :type fill: float
    :param axis: 0 or 1
    :type axis: int
    :return: data processed
    :rtype: pd.Dataframe
    """
    x = x0.copy()

    if axis == 0:
        x = x.T

    # 寻找上下界
    if bound_mode == 'std':
        lbv = x.mean(axis=1) + x.std(axis=1) * lb
        ubv = x.mean(axis=1) + x.std(axis=1) * ub
    elif bound_mode == 'quantile':
        lbv = x.quantile(lb, axis=1)
        ubv = x.quantile(ub, axis=1)
    else:
        lbv = x.max(x, axis=1)
        ubv = x.min(x, axis=1)

    if cut:
        # 界外值删除
        x[x.sub(lbv, axis=0) < 0] = fill
        x[x.sub(ubv, axis=0) > 0] = fill
    else:
        # 界外值缩尾
        sub_lb = x.sub(lbv, axis=0)
        x[sub_lb < 0] = x[sub_lb < 0] - sub_lb[sub_lb < 0]
        sub_ub = x.sub(ubv, axis=0)
        x[sub_ub > 0] = x[sub_ub > 0] - sub_ub[sub_ub > 0]

    if axis == 0:
        x = x.T

    return x


@_convert_multi_index(['x1'])
def _pn_quantile(x1):
    """
    计算某个值在所在截面上的分位数， by axis。计算方法：（排名-0.5）/非缺失值总数
    :param data: data to be calculated
    :type data: pd.Series(MultiIndex) or pd.Dataframe
    :param axis: 0 or 1
    :type axis: int
    :return: quantiles of data
    :rtype: pd.DataFrame
    """
    return __pn_quantile(x1)


@_convert_multi_index(['x1'])
def _pn_z_score(x1):
    """
    计算某个值在所在截面上的z_score， by axis.

    :param x1: x1 to be z_score
    :type x1: pd.Series(MultiIndex) or pd.Dataframe
    :param axis: 0 or 1
    :type axis: int
    :return: z_score
    :rtype: pd.Dataframe
    """
    data_pivot = x1.sub(x1.mean(axis=1), axis=0).div(x1.std(axis=1), axis=0)
    return data_pivot


@_convert_multi_index(['x1'])
def _pn_normal(x1):
    """
    在截面上做正态化处理。
    正态化处理：求一个向量中每个数字在向量中的分位数，然后将该分位数代入正态分布cdf的反函数中

    :param x1: x1 to be transformed to normal distribution
    :type x1: pd.Series(MultiIndex) or pd.Dataframe
    :return: normal distributed data
    :rtype: pd.DataFrame
    """
    data_pivot = __pn_quantile(x1)
    for date in data_pivot.index:
        data_pivot.loc[date] = norm.ppf(data_pivot.loc[date].values)
    return data_pivot


@_convert_multi_index(['x1'])
def _pn_winsorize(x1):
    """
    根据数据在所在截面上的 z-score（偏离均值的标准差倍数）（或分位数） 划出上下界，对超出上下界的数据进行缩尾处理（winsorize）
    bound_mode为'std'时，按z-score划分，lb、ub代表下界、上界的离均值的标准差倍数。
    bound_mode为'quantile'时，按分位数划分，lb、ub代表下界、上界的分位数。
    :param x: data to be cut
    :type x: pd.Series(MultiIndex) or pd.Dataframe
    :return: data processed
    :rtype: pd.Dataframe
    """
    data_pivot = __pn_cut(x1, bound_mode='std', lb=-3.0, ub=3.0, cut=False, axis=1)
    # data_pivot = data_pivot.stack()
    # data_pivot = data_pivot.reindex(x1.index)
    return data_pivot


@_convert_multi_index(['x1'])
def _ts_std(x1, win=240):
    """
    时序滚动标准差 by stock

    :param x1: data to be smoothed
    :type x1: pd.Series(MultiIndex) or pd.Dataframe
    :param win: rolling window size
    :type win: int
    :param min_periods: 计算标准差需要的最少数据窗口数。默认与win相同。
    :type min_periods: int or None(default)
    :param ddof: 计算标准差时的分母：n-ddof
    :type ddof: int
    :return: time-series rolling std of data
    :rtype: pd.DataFrame
    """
    ts = x1.rolling(win, min_periods=None).std(ddof=0)
    return ts


@_convert_multi_index(['x1'])
def _ts_quantile(x1, win=240):
    """
    时序滚动分位数。
    :param x1: 用于计算的数据
    :type x1: pd.Series(MultiIndex) or pd.Dataframe
    :param win: rolling window size
    :type win: int
    :return: 计算结果
    :rtype: pd.DataFrame
    """
    min_periods = win
    res = pd.DataFrame().reindex_like(x1)
    for i in range(min_periods - 1, x1.shape[0]):
        st, ed = max(0, i - win + 1), i + 1
        res.iloc[i, :] = __pn_quantile(x1.iloc[st:ed, :], axis=0).iloc[-1]

    return res


@_convert_multi_index(['x1'])
def _ts_decay(x1, win=5):
    """
    时序滚动平滑：(e.g. win=5, t-0~t-4: 5/15, 4/15, 3/15, 2/15, 1/15.)。
    若数据中存在缺失值，该数据会被忽略，其权重会被等比例分到其他日期的数据上。（权重之和必定为1）
    :param x1: data to be smoothed
    :type x1: pd.Series(MultiIndex) or pd.Dataframe
    :param win: rolling window size
    :type win: int
    :return: time-series rolling weighted mean of data0
    :rtype: pd.Dataframe
    """
    # 类型检测
    data = x1.values

    res = np.full_like(data, np.nan)
    tmp = np.full([win, data.shape[1]], np.nan)
    for i in range(0, data.shape[0]):
        tmp[0:win - 1, :] = tmp[1:win, :]
        tmp[-1, :] = data[i, :]
        w = (np.array(range(win)) + 1).reshape(-1, 1)
        np.seterr(divide='ignore', invalid='ignore')
        res[i, :] = np.nansum(tmp * w, axis=0) / np.nansum(~np.isnan(tmp) * w, axis=0)
        np.seterr(divide='warn', invalid='warn')

    return pd.DataFrame(res, index=x1.index, columns=x1.columns)


def _ts_shift(x1, win=1):
    with np.errstate(divide='ignore', invalid='ignore'):
        time_index_value = x1.unstack()
        time_index_value = time_index_value.shift(win).stack()
        time_index_value = time_index_value.reindex(x1.index)
        return time_index_value.fillna(0)

@_convert_multi_index(['x1'])
def _ts_mean(x1, win):
    time_index_value = x1.rolling(win).mean()
    return time_index_value


@_convert_multi_index(['x1', 'x2'])
def _ts_corr(x1, x2, win):
    """
    计算x1，x2中对应列的滚动时序相关性（x1第n列与x2第n列的相关性）。
    当x2只有一维时，将x2复制x1.shape[1]列，使x1与x2同shape
    可以通过mode参数对x2执行 'abs' (取绝对值)/'positive'(只取正值)/'negative'(只取负值)的处理

    :param x1: data #1 used to calculate time-series rolling correlation with data #2
    :type x1: pd.Series(MultiIndex) or pd.Dataframe
    :param x2: data #2 used to calculate time-series rolling correlation with data #1
    :type x2: pd.Series(MultiIndex) or pd.Dataframe or pd.series;
              if d1 is Series, duplicate to the same shape as x1
    :param win: rolling window size
    :type win: int
    :param min_periods: minimum window
    :type min_periods: int
    :param mode: 'normal' or 'abs' or 'positive' or 'negative' (apply on d1)
    :type mode: str
    :return: stock by stock time-series rolling correlation/covariance
    :rtype: pd.Dataframe
    """
    min_periods = max(1, int(win/4))
    return __ts_cov(x1, x2, win, min_periods, 'normal', is_cov=False)


@_convert_multi_index(['x1', 'x2'])
def _ts_cov(x1, x2, win):
    """
    计算x1，x2中对应列的滚动时序协方差（x1第n列与x2第n列的协方差）。
    当x2只有一维时，将x2复制x1.shape[1]列，使x2与x1同shape
    可以通过mode参数对x2执行'abs'(取绝对值)/'positive'(只取正值)/'negative'(只取负值)的处理

    :param x1: data #1 used to calculate time-series rolling correlation with data #2
    :type x1: pd.Series(MultiIndex) or pd.Dataframe
    :param x2: data #2 used to calculate time-series rolling correlation with data #1
    :type x2: pd.Series(MultiIndex) or pd.Dataframe or np.array(ndim=1,2) or pd.series;
              if ndim=1, duplicate to the same shape as x1
    :param min_periods: minimum window
    :type min_periods: int
    :param win: rolling window size
    :type win: int
    :param mode: 'normal' or 'abs' or 'positive' or 'negative' (apply on d1)
    :type mode: str
    :return: stock by stock time-series rolling correlation/covariance
    :rtype: pd.Dataframe
    """
    min_periods = max(1, int(win / 4))
    return __ts_cov(x1, x2, win, min_periods, 'normal', is_cov=True)


def __ts_cov(d0, d1, win, min_periods, mode='normal', is_cov=True):
    data0 = d0.values
    data1 = d1.values

    if (np.ndim(d1) == 1):
        data1 = data1.reshape(-1, 1)
        data1 = np.tile(data1, data0.shape[1])
    elif (data1.shape[1] == 1):
        data1 = data1.reshape(-1, 1)
        data1 = np.tile(data1, data0.shape[1])

    # d1的可选数据处理
    if 'abs' in mode:
        data1 = abs(data1)
    if 'positive' in mode:
        data1[data1 <= 0] = np.nan
    if 'negative' in mode:
        data1[data1 >= 0] = np.nan

    res = np.full_like(data0, np.nan)

    for i in range(min_periods - 1, data0.shape[0]):
        st, ed = max(0, i - win + 1), i + 1
        x = data0[st:ed, :]
        y = data1[st:ed, :]

        # 剔除全nan列以防弹出warning
        invalid_idx = (np.isnan(x) | np.isnan(y)).all(axis=0)
        if invalid_idx.all():
            continue
        idx = ~invalid_idx

        x = x[:, idx]
        y = y[:, idx]
        x_mean = np.nanmean(x, axis=0)
        y_mean = np.nanmean(y, axis=0)
        numerator = np.nanmean((x - x_mean) * (y - y_mean), axis=0)
        if not is_cov:
            std_x = np.nanstd(x, axis=0)
            std_x[std_x == 0] = np.nan
            std_y = np.nanstd(y, axis=0)
            std_y[std_y == 0] = np.nan
            numerator = numerator / (std_x * std_y)

        res[i, idx] = numerator
    return pd.DataFrame(res, index=d0.index, columns=d0.columns)


ts_corr = make_ts_function(function=_ts_corr, name='ts_corr', arity=2, wrap=False, is_ts=True, days=[2, 3, 4])
ts_cov = make_ts_function(function=_ts_cov, name='ts_cov', arity=2, wrap=False, is_ts=True, days=[2, 3, 4])
ts_decay = make_ts_function(function=_ts_decay, name='ts_decay', arity=1, wrap=False, is_ts=True, days=[2, 3, 4])
ts_std = make_ts_function(function=_ts_std, name='ts_std', arity=1, wrap=False, is_ts=True, days=[2, 3, 4])
ts_quantile = make_ts_function(function=_ts_quantile, name='ts_quantile', arity=1, wrap=False, is_ts=True, days=[2, 3, 4])
# ts_shift = make_ts_function(function=_ts_shift, name='ts_shift', arity=1, wrap=False, is_ts=True, days=[1, 3, 5, 20])
ts_mean = make_ts_function(function=_ts_mean, name='ts_mean', arity=1, wrap=False, is_ts=True, days=[2, 3, 4])

pn_winsorize = make_pd_function(function=_pn_winsorize, name='pn_winsorize', arity=1, wrap=False)
pn_normal = make_pd_function(function=_pn_normal, name='pn_normal', arity=1, wrap=False)
pn_quantile = make_pd_function(function=_pn_quantile, name='pn_quantile', arity=1, wrap=False)
pn_log = make_pd_function(function=_pn_log, name='pn_log', arity=1, wrap=False)
pn_inv = make_pd_function(function=_pn_inv, name='pn_inv', arity=1, wrap=False)
pn_div = make_pd_function(function=_pn_div, name='pn_div', arity=2, wrap=False)

# 无法通过校验，以后改规则
pn_z_score = _Function(function=_pn_z_score, name='pn_z_score', arity=1)

pn_add = _function_map['add']
pn_sub = _function_map['sub']
pn_mul = _function_map['mul']
pn_sqrt = _function_map['sqrt']
pn_abs = _function_map['abs']
pn_neg = _function_map['neg']
pn_max = _function_map['max']
pn_min = _function_map['min']
pn_sin = _function_map['sin']
pn_cos = _function_map['cos']
pn_tan = _function_map['tan']
