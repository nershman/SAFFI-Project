# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-11 13:00:46
# @Last Modified by:   sma
# @Last Modified time: 2021-05-11 22:05:48

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

from search_engines import Google
import facebook_scraper as fs
from netmums import scrapehelpers as scr #requires making path visible
from netmums import scrape_netmums_basic as nm

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


search_queries = scr.make_combo_list(['site:facebook.com/*/posts/'+' baby'],products, hazards)
#without or with a slash on the end doesnt matter.
search_queries = search_queries[0:2] #DEBUG #DELETE!!!
#get the URL of first 10 pages of results for each query
num_pages = 10
results_dict = {}
for query in search_queries:
	engine.results.__init__() #reset the results object (o.w. it keeps results from previous queries)
	search_results = engine.search(query, pages=num_pages).links() #get links
	results_dict[query] = ['link':result for result in search_results] # convert to list of dicts.

#convert to url_dict (which will remove duplicates)
results_dict = nm.resultsdict_to_urldict(results_dict)

#TODO separate into pages vs individual posts.

#TODO: save a pickle. (remember to close the file or whatever!)

###################################################
## use facebook scraper to get info ##
###################################################

#save to csv (or pkl it if its enormopus as csv..)
#remember we want to collect the comments.
#so using pickle might be better...,.