#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 19:07, 04/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from models.utils.dataset_util import make_train_and_test_datafile
from config import Config

# =========================== Making training and testing dataset for both open-source data and proprietary data

# Linux
output_linux = [Config.LINUX_TRAIN_TEST, Config.LINUX_TRAIN, Config.LINUX_TEST]
make_train_and_test_datafile(Config.LINUX_TEMPLATES, output_linux, Config.LINUX_TEST_SIZE)

# Windows
output_wins = [Config.WINDOWS_TRAIN_TEST, Config.WINDOWS_TRAIN, Config.WINDOWS_TEST]
make_train_and_test_datafile(Config.WINDOWS_TEMPLATES, output_wins, Config.WINDOWS_TEST_SIZE)

