# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-11 13:00:46
# @Last Modified by:   sma
# @Last Modified time: 2021-05-22 15:00:51

"""
First Approach:
1) search google for posts
	- this will guarantee more relevant topics, and more likely to return high-traffic posts (more comments)
2) feed post URLs into facebook scraper
	- goals: keep query source as previously w netmums scrape.
"""

import os #DEBUG
os.chdir('/Users/sma/Documents/INRAE internship/scrape-git/facebook') #DEBUG

import sys
sys.path.append('../') #make parent path visible so we can import modules from other folders.

from search_engines import Startpage
import facebook_scraper
from netmums import scrapehelpers as scr #requires making parent path visible
from facebook import facebook_helper as facehelp
import pickle 
import time
import re
from random import uniform as random_uniform

################
##  set up	##
################

#in case we need to re-run to capture more links, we can easily update the file names from here.
filenames = ['sp-fb_search_results.pkl',
'sp-fb_safety.pkl',
'sp-deadURLs.txt',
'sp-fb_data_searchscrape.pkl',
'sp_search_results_temp.pkl']

###################################################
## use scraper to get links to posts on facebook ##
###################################################

# build combo list

hazards = scr.get_concerns()
products = scr.get_foods()

#modify them to get better results 
#by using quotes.
hazards = ["\""+i+"\"" for i in hazards]
hazards.append('\"recall\"')
hazards.append('\"product recall\"')

products = ["\""+i+"\"" for i in products]
products[4] = "\"veggie\" \"baby food\""
products[5] = "\"vegetable\" \"baby food\""
products[6] = "\"veg\" \"puree\""
products[7] = "\"veg\" \"purée\""
products[8] ="\"fruit\" \"puree\""
products[9] ="\"fruit\" \"baby food\""
products[10] ="\"fruit\" \"purée\""
products.append('\"recall\"')
products.append('\"product recall\"')

#create combo list
#we add baby to it just to get more relevant results, it's not required since it's not in quotes.
#we also add our URl restriction as a string.

#this string returns either pages or posts on pages.
#* site:facebook.com/*/posts "baby food" arsenic
#* site:facebook.com/groups/* "baby food" "arsenic"
#	* this one only gets like 4 pages of results on google.

search_queries = scr.make_combo_list(['site:en-gb.facebook.com/*/posts/ OR site:www.facebook.com/*/posts/'+' baby'],
										products, hazards)
#without or with a slash on the end doesnt matter.
#en-gb.facebook, en-us.facebook, www.facebook 

#search_queries = search_queries[0:2] #DEBUG #DELETE!!!


my_header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5)\
            AppleWebKit/537.36 (KHTML, like Gecko) Cafari/537.36'}

#get the URL of first 10 pages of results for each query
engine = Startpage()
engine.set_headers(my_header)
num_pages = 10
results_dict = {}


for query in search_queries:
	if query not in results_dict:
		print(len(results_dict), "/", len(search_queries), " completed")
		try:
			engine.results.__init__() #reset the results object (o.w. it keeps results from previous queries)
			engine.search(query, pages=num_pages).links() #get links

		except KeyboardInterrupt: #if you press ctrl-c in the console now, the process will end 
			exit()
