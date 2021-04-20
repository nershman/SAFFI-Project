# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-04-19 14:57:15
# @Last Modified by:   sma
# @Last Modified time: 2021-04-19 15:14:09

from itertools import product

def make_combo_list(*args, sep=' '):
	"""
	Takes any number of lists of strings and returns a list of all combinations.
	optional argument sep to change how strings are combine(e.g use + for search
	URLS)
	"""
	for item in args:
		if isinstance(item, list):
			pass
		else:
			raise TypeError('argument contains non-list object')
			break
		if all(isinstance(val, str) for val in item):
			pass
		else:
			raise TypeError('item in list is not of type string')
			break
	return [sep.join(i) for i in list(product(*args))]


problems = ["Chemical contaminants",
"Endocrine disruptor",
"preservatives",
"sweeteners",
"additives",
"Pesticides",
"Veterinary drugs",
"GMO",
"Metals",
"Mycotoxin",
"Bisphenol A",
"Furan and furan-like molecules",
"DON",
"Dioxin and PCB",
"MOH",
"Mineral oil hydrocarbons",
"MOSH",
"MOAH",
"Nitrates",
"Acrylamid",
"Phtalates",
"Microbiologic contaminants",
"Salmonella",
"Campylobacter",
"Listeria",
"EColi",
"Cronobacter",
"Histamine",
"Other bacteria",
"Virus",
"Parasites"]


foods = [
#infant formula
"formula","baby formula",
#sterizlized vegetable mixed with fish
"veggie baby food","vegetable baby food",
#fresh fruit puree mildly processed
"fruit puree","fruit baby food",
#infant cereals
"cereal for baby", "cereal"]


