# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-11 13:00:46
# @Last Modified by:   sma
# @Last Modified time: 2021-05-16 11:59:17

"""
First Approach:
1) search google for posts
	- this will guarantee more relevant topics, and more likely to return high-traffic posts (more comments)
2) feed post URLs into facebook scraper
	- goals: keep query source as previously w netmums scrape.
"""

#import os #DEBUG
#os.chdir('/Users/sma/Documents/INRAE internship/scrape-git/facebook') #DEBUG

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

################
##  set up	##
################

#in case we need to re-run to capture more links, we can easily update the file names from here.
filenames = ['few-fb_search_results.pkl',
'few-fb_safety.pkl',
'few-deadURLs.txt',
'few-fb_data_searchscrape.pkl',
'few-google_search_results_temp.pkl']

###################################################
## use scraper to get links to posts on facebook ##
###################################################

# build combo list

products = 'baby formula OR bottle-fed OR veggie OR vegetable OR baby food OR veg puree OR fruit puree OR fruit food OR applesauce OR cereal OR  porridge OR oats OR oatmeal OR jar food OR baby food OR  premade OR puree OR  purée OR yoghurt OR pudding'
hazards = scr.get_concerns()

hazards = ["\""+i+"\"" for i in hazards]
hazards.append('\"recall\"')
hazards.append('\"product recall\"')

search_queries = scr.make_combo_list(['site:en-gb.facebook.com/*/posts/ OR site:www.facebook.com/*/posts/'],
									  hazards, [products])


#without or with a slash on the end doesnt matter.
#en-gb.facebook, en-us.facebook, www.facebook 

#search_queries = search_queries[0:2] #DEBUG #DELETE!!!

header_strings = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:53.0) Gecko/20100101 Firefox/53.0',
'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36 Edge/14.14393',
'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)',
'Mozilla/5.0 (Windows; U; MSIE 7.0; Windows NT 6.0; en-US)',
'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 1.1.4322; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)',
'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0;  Trident/5.0)',
'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Trident/6.0; MDDCJS)',
'Mozilla/5.0 (compatible, MSIE 11, Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko',
'Mozilla/5.0 (iPad; CPU OS 8_4_1 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Version/8.0 Mobile/12H321 Safari/600.1.4',
'Mozilla/5.0 (iPhone; CPU iPhone OS 10_3_1 like Mac OS X) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.0 Mobile/14E304 Safari/602.1',
'Mozilla/5.0 (Linux; Android 6.0.1; SAMSUNG SM-G570Y Build/MMB29K) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/4.0 Chrome/44.0.2403.133 Mobile Safari/537.36',
'Mozilla/5.0 (Linux; Android 5.0; SAMSUNG SM-N900 Build/LRX21V) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/2.1 Chrome/34.0.1847.76 Mobile Safari/537.36',
'Mozilla/5.0 (Linux; Android 6.0.1; SAMSUNG SM-N910F Build/MMB29M) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/4.0 Chrome/44.0.2403.133 Mobile Safari/537.36',
'Mozilla/5.0 (Linux; U; Android-4.0.3; en-us; Galaxy Nexus Build/IML74K) AppleWebKit/535.7 (KHTML, like Gecko) CrMo/16.0.912.75 Mobile Safari/535.7',
'Mozilla/5.0 (Linux; Android 7.0; HTC 10 Build/NRD90M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.83 Mobile Safari/537.36']
#my_header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5)\
#            AppleWebKit/537.36 (KHTML, like Gecko) Cafari/537.36'}
#new
#my_header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:83.0) Gecko/20100101 Firefox/83.0'}
#my_header={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36 Edge/14.14393'}


my_header={'User-Agent':header_strings[9]}
#get the URL of first 10 pages of results for each query
engine = Google()
engine.set_headers(my_header)
num_pages = 3
results_dict = {}

del search_queries[13] #DEBUG DELETE
del search_queries[14] #DEBUG DELETE
#get the existing results
try:
	with open(filenames[4], "rb") as output:
		results_dict = pickle.load(output)
except:
	#if theres no file then we need to save one.
	with open(filenames[4], "wb") as output:
		pickle.dump(results_dict, output)

block_count = 0

for query in search_queries:
	if query not in results_dict:
		print(len(results_dict), "/", len(search_queries), " completed")
		time.sleep(random_uniform(2,7))
		try:
			engine.results.__init__() #reset the results object (o.w. it keeps results from previous queries)
			search_results = engine.search(query, pages=num_pages).links() #get links


			if not engine.is_banned:
				results_dict[query] = [{'link':result} for result in search_results] # convert to list of dicts.

				#save the file again, since we don't think there were any errors.
				print("SAVING...")
				with open(filenames[4], "wb") as output:
					pickle.dump(results_dict, output)
				print("done.")
			else:
				if block_count == 1:
					print('resetting cookies didnt work, exiting')
					exit()
				print("banned, resetting cookies.")
				block_count += 1
				engine._http_client = search_http_client.HttpClient()
				
		except KeyboardInterrupt: #if you press ctrl-c in the console now, the process will end 
			exit()

		if not search_results or search_results is None:
			#when theres no results it goes too fast, slow it down.
			print(query + ': no results. :(')
			time.sleep(random_uniform(8,27))


engine._http_client

#convert to url_dict (which will remove duplicates)
url_dict = facehelp.resultsdict_to_urldict(results_dict)

#drop keys which don't conform to facebook.com and posts. (just in case)
badkeys = []
for key in url_dict.keys():
	if 'facebook.com' not in key:
		badkeys.append(key)
	if '/posts' not in key:
		badkeys.append(key)
for key in set(badkeys):
	url_dict.pop(key)

print('\n \n search results retrieved, saving to file... \n \n ')
#save a pickle
filehandler = open(filenames[0], 'wb')  
pickle.dump(url_dict, filehandler)
filehandler.close()


#NOTES
# if google starts denying without returning an error (so all requests get no results)
# we can look at the list. since I am iterating over a list
# it will be in the same order each time.
# so I can pop all keys after the one where it went wrong (which I can find from the KERNEL OUTPUT)

#if all searchers return no results, try setting up a new user agent, and/or (pref. both!) changin to a different proxy

#if google starts giving 429 or other denials, it seems pretty hopeless AFAIK. maybe if i cahnge user agent agian.

