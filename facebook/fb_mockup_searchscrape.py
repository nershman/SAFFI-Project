# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-11 13:00:46
# @Last Modified by:   sma
# @Last Modified time: 2021-05-14 13:54:08

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
import facebook_scraper
from netmums import scrapehelpers as scr #requires making parent path visible
from facebook import facebook_helper as fb
import pickle 
import time
import re
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
url_dict = fb.resultsdict_to_urldict(results_dict)

#drop keys which don't conform to facebook.com and posts. (just in case)
badkeys = []
for key in new_dictionary.keys():
	if 'facebook.com' not in key:
		badkeys.append(key)
	if '/posts' not in key:
		badkeys.append(key)
for key in set(badkeys):
	new_dictionary.pop(key)

#TODO; doubel check that its working and keeping the queries or whatever
#at thsi point the data structure should be a ulr as key, where the url is that of a facebook thing. the value 
# is a dict which contains the key 'query' with values corresponding to a set of queries.

#TODO: save a pickle. (remember to close the file or whatever!)
filehandler = open('facebook_results.pkl', 'wb')  
pickle.dump(final_data, filehandler)
filehandler.close()


###################################################
## use facebook scraper to get info ##
###################################################

#based on the URL, categorize keys into two lists for posts vs pages.
pages = []
posts = []
for key in new_dictionary.keys():
	if re.search('facebook.com\/([^\/]+)\/posts\/(.+)', key):
		#add to posts if the URL contains stuff after /posts/
		posts.append(key)
	else:
		pages.append(key)

#TODO 
#this is just pseudocode lol
for url in posts:
	#the value for each key shoudl already be a dict containing query words.
	urldict[url]['data'] =  facebook_scraper.

for url in pages:
	#do the same but scrape first n posts from the page (50??)

#TODO: make sure that we collect comments on posts.

#save to pkl




#########

#my facebook options.
my_fb_options={"posts_per_page": 200, "comments":True, "reactors": True}

pages = 10
timeout = 45
