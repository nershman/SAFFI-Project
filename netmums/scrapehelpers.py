# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-04-19 14:57:15
# @Last Modified by:   sma
# @Last Modified time: 2021-05-07 09:37:20

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


def get_concerns():
	return ["Chemical contaminants",
#ENDOCRINE DISRUPTOR
"Endocrine disruptor",
"endocrine",#end
#FOOD PRESERVATIVES, SWEETENERS AND ADDITIVES
"preservatives",
"sweeteners",
"additives", #end
"Pesticides",
#VETERINARY DRUGS
"Veterinary drugs",
"animal drugs",
"vet drugs", #end
#GMO
"GMO",
"GM",
"genetically modified",#end
"Metals",
"Mycotoxin",
#BISPHENOL A
"Bisphenol A",
"BPA", #end
#FURAN - removed because nothing related to this returns results
#DON (note that this acronym nobody uses and all results are from words like "don't")
"deoxynivalenol",
"vomitoxin",#end
#DIOXIN AND PCB
"Dioxin",
"PCB",
"biphenyls",#end
#MOH
"MOH",
"hydrocarbons",
"saturdated hydrocarbons",
"MOAH",
"aromatic hydrocarbons",#end
"Nitrates",
#ACRYLAMID
"Acrylamid",
"Acrylamide",
"phthalates",
#MICROBIOLOGIC CONTAMINANTS
"Microbiologic contaminants",
"bacteria",
"spores",
"mold",
"mould",
"virus",
"microbes",
"contaminated",#end
"Salmonella",
"Campylobacter",
"Listeria",
#ECOLI
"EColi",
"E-coli", #end
"Cronobacter",
"Histamine",
#other bacteria
"bacteria",#end
"Virus",
"Parasites",
#UNRELATED BUT MAYBE USEFUL?
"carcinogen",
"chemicals", 
"toxic", 
"toxin", 
"poisonous", 
"fungus", 
"food poisoning", 
"hazard"]


def get_foods():
	return [
#infant formula
"formula","baby formula", "bottle-fed", "bottle",
#sterizlized vegetable mixed with fish
"veggie baby food","vegetable baby food",
"veg puree", "veg purée",
#fresh fruit puree mildly processed
"fruit puree","fruit baby food", "fruit purée", "applesauce",
#infant cereals
"cereal for baby", "cereal", "porridge", "oats", "oatmeal",
#other
"jar food", "baby food", "jarred", "premade food", "puree", "purée", "jarred food"
,"yoghurt", "pudding"]


