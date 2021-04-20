# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-04-19 15:01:45
# @Last Modified by:   sma
# @Last Modified time: 2021-04-19 15:01:45

#test combo list
from itertools import repeat, permutations
l1 = ["a", "b"]
l2 = ["c", "d"]

temp = list(product(l1,l2,l2))
make_combo_list(l1,l2)
make_combo_list(l1,["c","d",2])
make_combo_list(l1,2)
make_combo_list(l1,"oxo")