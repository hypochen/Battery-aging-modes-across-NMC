# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:05:18 2021

@author: WALKCM2
"""

# Removes RPTs by treating them as seasonal effects and removing them with Seasonal Decomposition of Time Series with period = 50

import statsmodels.api as sm

def detrendCBCdict(pack_dict, period=50):
    for pack in pack_dict:
        for cell in pack_dict[pack]:
            for measurement in pack_dict[pack][cell]:
                res = sm.tsa.seasonal_decompose(pack_dict[pack][cell][measurement],period=period,extrapolate_trend = 'freq')
                pack_dict[pack][cell][measurement] = res.trend
                    
    return pack_dict