#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:57, 17/06/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import choice, seed
from numpy import array


def create_sample_index(rate, bound_range):
    # seed(13)
    if isinstance(rate, int):
        return choice(bound_range, rate, replace=False)
    else:
        return choice(bound_range, int(rate*bound_range), replace=False)


def sample_arrays(arrays, index):
    ret = []
    for a in arrays:
        if a is None:
            ret.append(None)
            continue
        if type(a) is array:
            ret.append(a[index])
        if type(a) is list:
            selected_data = []
            for idx in index:
                selected_data.append(a[idx])
            ret.append(selected_data)
    return tuple(ret)
