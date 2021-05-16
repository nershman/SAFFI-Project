# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-16 16:08:44
# @Last Modified by:   sma
# @Last Modified time: 2021-05-16 17:39:37
"""
This script takes a path of a directory containing html pages of google results,
with a format where the name is a two digit or one digit number.
"""
import sys
sys.path.append('../') #make parent path visible so we can import modules from other folders.

from search_engines import Google
from search_engines import http_client as search_http_client
import facebook_scraper
from netmums import scrapehelpers as scr #requires making parent path visible
from facebook import facebook_helper as facehelp
import pickle 
import time
import re
from random import uniform as random_uniform
import os

from bs4 import BeautifulSoup

scrape_path = '/Users/sma/Documents/INRAE internship/google_search_by_hand'


engine = Google() #we use the package to parse the html links

results_dict = {}
for filename in os.listdir(scrape_path):
	if filename == '.DS_Store':
		pass
	else:

		engine.results.__init__() #reset the list inside engine object
	
		with open(scrape_path+'/'+filename, 'r') as file:
			temp = file.readlines()
	
		temp = ' '.join(temp) #make lsit of strings to one string.
	
		tags = BeautifulSoup(temp, "html.parser")
		items = engine._filter_results(tags)
		engine._collect_results(items) #append the results to the list inside engine object.
		
		#add the list to our dict.
		results_dict[filename] = engine.results.links()
	
		if not results_dict[filename]:
			print('something went wrong with ', filename, 'no results ...')
	
#flatten stuff...

results_dict = {key[0:2]: [item for k, val in results_dict.items() for item in val if k[0:2] == key[0:2]]for key,value in results_dict.items()}

#replace with the original search queries.
products = 'baby formula OR bottle-fed OR veggie OR vegetable OR baby food OR veg puree OR fruit puree OR fruit food OR applesauce OR cereal OR  porridge OR oats OR oatmeal OR jar food OR baby food OR  premade OR puree OR  purée OR yoghurt OR pudding'
hazards = scr.get_concerns()

hazards = ["\""+i+"\"" for i in hazards]
hazards.append('\"recall\"')
hazards.append('\"product recall\"')

search_queries = scr.make_combo_list(['site:en-gb.facebook.com/*/posts/ OR site:www.facebook.com/*/posts/'],
									  hazards, [products])

#rename keys based on teh value in the list
for number in results_dict.keys():
	results_dict[search_queries[int(number) - 1]] = results_dict.pop(number)

#save to pickle
with open('manual_search_resdict.pkl','wb') as f:
	pickle.dump(results_dict, f)
