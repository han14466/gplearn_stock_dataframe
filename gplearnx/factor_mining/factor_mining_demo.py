#!/usr/bin/env python  
# -*- coding:utf-8 _*-

from cmath import nan
import math
import pandas as pd
import numpy as np
import datetime
from gplearnx.factor_mining.enum_type import PeriodType
from gplearnx.fitness import make_fitness

from utils.calendar import ChinaCalendar

from definition.universe import BasicUniverse, STUniverse, NewListedUniverse, RemoveUniverse, IndexUniverse
from utils.factor_processing import (
    StandardizationFactory,
    ExtremeFactory,
    NeutralizeFactory
)
from factor.factor_analysis.ic import _calc_ic_term_structure
from gplearnx.factor_mining.period_util import PeriodUtil
from gplearnx.functions_x import pn_add, pn_sub, pn_mul, pn_div, pn_sqrt, \
    pn_abs, pn_neg, pn_max, pn_min, pn_sin, pn_cos, pn_tan, \
    pn_winsorize, pn_normal, pn_z_score, pn_quantile, pn_log, pn_inv, \
    ts_cov, ts_corr, ts_decay, ts_std, ts_quantile, ts_mean, \
    _pn_z_score
from factor.factor_analysis.ic import IC
from gplearnx.genetic import SymbolicTransformerX
from gplearnx.factor_mining.ray_utils import init_ray


calendar = ChinaCalendar.get_instance()


def get_returns(start_date, end_date):
    returns = dataapi.get_stock_return(field="ChangePCT", start_date=start_date, end_date=end_date) / 100
    return returns


def annual_multiplier(factor_value):
    rebalance_dates = factor_value.index.unique('trade_time').sort_values()
    start_date = rebalance_dates[0].date()
    end_date = rebalance_dates[-1].date()
    cal = ChinaCalendar.get_instance()
    dates = cal.get_trading_dates(start_date, end_date)
    n = len(rebalance_dates) / len(dates) * 250
    _annual_multiplier = np.sqrt(n)
    return _annual_multiplier


def rank_ic2(data, returns, long_ic=True):
    """计算rank ic"""
    rank_ic = IC(data, returns, period=1, method='normal', min_term=1, debug=False, decay_terms=20)
    ic_data = rank_ic.ic_data
    return ic_data


def post_process_data(factor_value):
    factor_series = factor_value.copy()
    factor_series = factor_series.replace([np.inf, -np.inf], np.nan)
    factor_series = factor_series.dropna()
    factor_series = ExtremeFactory.mad(factor_series)
    # factor_series = ExtremeFactory.median_nsigma(factor_series, num=3)

    # 标准化
    factor_series = StandardizationFactory.standard(factor_series)
    # 市值行业中性化
    factor_series = NeutralizeFactory.industry_market_neutralize(factor_series, value_type='TotalMV',
                                                                 industry_type='sw_FirstIndustryCode')
    # 标准化
    factor_series = StandardizationFactory.standard(factor_series)
    return factor_series


def filter_universe(start_date, end_date):
    ru = RemoveUniverse(BasicUniverse(), NewListedUniverse(180), STUniverse())
    universe_code = ru.get_universe_series(start_date=start_date, end_date=end_date)
    return universe_code


def get_market_data(start_date, end_date, market_fields, period_type):
    """获取x_train"""
    data = dataapi.get_stock_field(fields=market_fields, start_date=start_date, end_date=end_date)

    universe_code = filter_universe(start_date, end_date)
    data = data.reindex(universe_code.index)

    for column_name in data.colums:
        data[column_name] = _pn_z_score(data[column_name])

    return data


def _ranck_icir_fitness(y, y_pred, w, **kwargs):
    if isinstance(y, np.ndarray) and isinstance(y_pred, np.ndarray):
        if np.array_equal(y, np.array([1, 1])) and np.array_equal(y_pred, np.array([2, 2])):
            return 1
        else:
            raise ValueError('function should not enter _my_metric_backtest this code')
    if isinstance(y_pred, np.ndarray):
        # y_pred.empty
        print(f"********************should not enter y_pred ndarray {y_pred}*****************************************")
        return 0
    start_date = y_pred.index.unique('trade_time').min()
    end_date = y_pred.index.unique('trade_time').max()
    if start_date is None or end_date is None:
        print(f"__________________Neither `start`{start_date} nor `end`{end_date} can be NaT----------------------")
        print(y_pred.head())
    else:
        factor_score = y_pred
    func_expression = kwargs.pop("func_expression", ' ')
    # print(f"current func_expression:{func_expression}")
    ic_mean = nan
    ann_icir = nan
    try:
        factor_value = y_pred.copy()
        factor_value = post_process_data(factor_value)

        index = factor_value.index.get_level_values(0).drop_duplicates().to_list()
        trade_data_list = PeriodUtil().get_period_trade_data_list(period_type, min(index), max(index))
        factor_value = factor_value.loc[trade_data_list]

        if lag > 0:
            factor_value = factor_value.copy()
            cal = ChinaCalendar.get_instance()
            factor_value = cal.shift(factor_value, lag)
        daily_returns = y.copy()
        factor_value.name = "factor_value"
        daily_returns.name = "daily_returns"
        ic = rank_ic2(factor_value, daily_returns)
        multiplier = annual_multiplier(factor_value)
        ic_mean = ic.mean()["IC"]
        ic_std = ic.std()["IC"]
        with np.errstate(divide='ignore', invalid='ignore'):
            icir = ic_mean / ic_std
            # icir = pd.Series(np.where(np.abs(ic_std) > 0.001, np.divide(ic_mean, ic_std), 1.), index=ic_mean.index)

        ann_icir = icir * multiplier
        print(f"ic:{ic_mean}, ir:{ann_icir}, func_expression:{func_expression}")
    except Exception as e:
        print(e)
        pass
    if ic_mean is nan or math.isnan(ic_mean):
        ic_mean = -10
    return ic_mean


rank_icir_fitness = make_fitness(function=_ranck_icir_fitness, greater_is_better=True)


if __name__ == '__main__':
    start_time = datetime.date(2017, 3, 1)
    end_time = datetime.date(2021, 3, 10)
    period_type = PeriodType.周度调仓
    lag = 1
    market_fields = 'PrevClosePrice;OpenPrice;HighPrice;LowPrice;ClosePrice;TurnoverVolume;TurnoverValue;ChangePCT;' \
                    'RangePCT;TurnoverRate;AvgPrice;TurnoverValueRW;TurnoverVolumeRW;' \
                    'ChangePCTRW;RangePCTRW;TurnoverRateRW;AvgPriceRW;HighPriceRW;LowPriceRW;' \
                    'HighestClosePriceRW;LowestClosePriceRW;TurnoverValuePerDayRW;TurnoverRatePerDayRW;' \
                    'TurnoverValueTW;TurnoverVolumeTW;ChangePCTTW;RangePCTTW;TurnoverRateTW;' \
                    'AvgPriceTW;HighPriceTW;LowPriceTW;HighestClosePriceTW;LowestClosePriceTW;' \
                    'TurnoverValuePerDayTW;TurnoverRatePerDayTW;TurnoverValueRM;TurnoverVolumeRM;' \
                    'TurnoverRateRM;AvgPriceRM;HighPriceRM;LowPriceRM;' \
                    'HighestClosePriceRM;LowestClosePriceRM;TurnoverValuePerDayRM;' \
                    'TurnoverRatePerDayRM;TurnoverValueTM;TurnoverVolumeTM;' \
                    'TurnoverRateTM;AvgPriceTM;HighPriceTM;LowPriceTM;HighestClosePriceTM;' \
                    'LowestClosePriceTM;TurnoverValuePerDayTM;TurnoverRatePerDayTM;' \
                    'TurnoverValueRMThree;TurnoverVolumeRMThree;TurnoverRateRMThree;' \
                    'TurnoverRateRMSix;TurnoverValueRY;TurnoverVolumeRY;' \
                    'TurnoverRateRY;AvgPriceRY;HighPriceRY;LowPriceRY;HighestClosePriceRY;' \
                    'LowestClosePriceRY;TurnoverValuePDayRY;TurnoverRatePDayRY;TurnoverValueYTD;' \
                    'TurnoverVolumeYTD;TurnoverRateYTD;AvgPriceYTD;' \
                    'HighPriceYTD;LowPriceYTD;HighestClosePriceYTD;LowestClosePriceYTD;TurnoverValuePerDayYTD;' \
                    'TurnoverRatePerDayYTD;HighAdjustedPrice;HighAdjustedPriceDate;' \
                    'LowAdjustedPrice;LowAdjustedPriceDate;' \
                    'YearVolatilityByDay;YearVolatilityByWeek;TotalMV;NegotiableMV;' \
                    'BackwardPrice;RisingUpDays;FallingDownDays;MaxRisingUpDays;MaxFallingDownDays;' \
                    'OpenPrice_hfq;HighPrice_hfq;LowPrice_hfq;ClosePrice_hfq'

    log_fields = "TurnoverVolume;TurnoverValue;TurnoverValueRW;TurnoverVolumeRW;" \
                 "TurnoverValuePerDayRW;TurnoverValueTW;TurnoverValueRM;TurnoverVolumeRM;" \
                 "TurnoverValuePerDayRM;TurnoverValueTM;TurnoverVolumeTM;TurnoverValuePerDayTM;" \
                 "TurnoverVolumeRMThree;TurnoverVolumeRMSix;TurnoverValueRY;TurnoverVolumeRY;" \
                 "TurnoverValuePDayRY;TurnoverValueYTD;TotalMV;NegotiableMV"

    z_score_flag = True
    train_data = get_market_data(start_time, end_time, market_fields, period_type)
    # 将train_data所有列变为float
    train_data = train_data.apply(pd.to_numeric)
    print(train_data.head())

    # print(f'isna count:{x_train.isna().sum()}')
    # train_data = x_train.dropna()

    y_train = get_returns(start_time, calendar.shift_date(train_data.index.unique('trade_time').max(), lag))
    print(y_train.head())

    x_train = train_data.dropna()

    init_function = [pn_add, pn_sub, pn_mul, pn_sqrt, pn_abs, pn_neg, pn_sin, pn_max, pn_min, pn_cos, pn_tan] + \
                    [ts_decay, ts_std, ts_mean]
    #ts_cov, ts_corr,ts_quantile, pn_quantile

    function_set = init_function

    # population_size = 10000
    # hall_of_fame = 2000
    # tournament_size = 2000
    # n_components = 50
    # generations = 4
    # random_state = 5

    population_size = 3000
    hall_of_fame = 1000
    tournament_size = 1000
    n_components = 100
    generations = 3
    random_state = 5

    module_list = ["/data/dev_workspace/gplearnx/gplearnx"]
    init_ray(module_list=module_list)

    est_gp = SymbolicTransformerX(
        feature_names=market_fields.split(';'),
        function_set=function_set,
        generations=generations,
        init_depth=(2, 4),
        metric=rank_icir_fitness,
        population_size=population_size,
        tournament_size=tournament_size,
        random_state=random_state,
        verbose=2, hall_of_fame=hall_of_fame,
        parsimony_coefficient=0.0001,
        p_crossover=0.4,
        p_subtree_mutation=0.01,
        p_hoist_mutation=0,
        p_point_mutation=0.01,
        p_point_replace=0.4,
        n_components=n_components,
        stopping_criteria=60,
        n_jobs=1)

    est_gp.fit(x_train, y_train)
    print(est_gp)
    best_programs_dict = {}
    for p in est_gp._best_programs:
        factor_name = 'alpha_' + str(est_gp._best_programs.index(p) + 1)
        best_programs_dict[factor_name] = {'fitness': p.fitness_, 'expression': str(p), 'depth': p.depth_,
                                           'length': p.length_}
    print(best_programs_dict)

    # with open(filename, 'rb') as f:
    #     est_gp = cloudpickle.load(f)
    #
    # best_programs_dict = {}
    # for p in est_gp._best_programs:
    #     factor_name = 'alpha_' + str(est_gp._best_programs.index(p) + 1)
    #     best_programs_dict[factor_name] = {'fitness': p.fitness_, 'expression': str(p), 'depth': p.depth_,
    #                                        'length': p.length_}
    # print(best_programs_dict)

    print("################################################")


    # for program in est_gp._best_programs:
    #     c_formula = program.__str__()
    #     program.raw_fitness(x_train, y_train, None)
    import ray
    x_train_ref = ray.put(x_train)
    y_train_ref = ray.put(y_train)
    future_ids = [program.raw_fitness(x_train_ref, y_train_ref, None) for program in est_gp._best_programs]
    ray.get(future_ids)

    print("============================================================")
    start_time = datetime.date(2021, 4, 1)
    end_time = datetime.date(2022, 4, 20)
    x_train = get_market_data(start_time, end_time, market_fields, period_type)
    y_train = get_returns(start_time, calendar.shift_date(x_train.index.unique('trade_time').max(), lag))

    x_train = x_train.dropna()


    x_train_ref = ray.put(x_train)
    y_train_ref = ray.put(y_train)
    future_ids = [program.raw_fitness(x_train_ref, y_train_ref, None) for program in est_gp._best_programs]
    ray.get(future_ids)

    print("finish")
    # """
