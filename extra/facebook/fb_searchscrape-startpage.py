# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-11 13:00:46
# @Last Modified by:   sma
# @Last Modified time: 2021-05-15 14:45:57

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


#get the existing results
try:
	with open(filenames[4], "rb") as output:
		results_dict = pickle.load(output)
except:
	#if theres no file then we need to save one.
	with open(filenames[4], "wb") as output:
		pickle.dump(results_dict, output)

for query in search_queries:
	if query not in results_dict:
		print(len(results_dict), "/", len(search_queries), " completed")
		time.sleep(random_uniform(1,4))
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
				print("quitting due to ban")
				exit()
		except KeyboardInterrupt: #if you press ctrl-c in the console now, the process will end 
			exit()

		if not search_results or search_results is None:
			#when theres no results it goes too fast, slow it down.
			print(query + ': no results. :(')
			time.sleep(random_uniform(2,6))




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


###################################################
## use facebook scraper to get info ##
###################################################

#based on the URL, categorize keys into two lists for posts vs pages.
pages = []
posts = []
for key in url_dict.keys():
	if re.search('facebook.com\/([^\/]+)\/posts\/([^\?]+)', key):
		#add to posts if the URL contains stuff after /posts/
		posts.append(key)
	else:
		pages.append(key)

#keep track of error URLs
dead_urls = []

print('\n beginning FB scrape... \n ')

###my facebook options.###
my_fb_options = {"posts_per_page": 10, "comments":True, "reactors": False, "reactions":True}
cookie_path = '/Users/sma/Documents/INRAE internship/scrape_files_confidential/fb_cookies.txt'
num_fb_pg = 4
fb_timeout = 45
stuff_we_dont_need = {'video', 'image_lowquality', 'images', 'image', 
						'is_live', 'shared_post_id', 'shared_post_url', 
						'video', 'video_thumbnail', 'video_duration_seconds', 
						'video_height', 'video_id', 'video_quality', 
						'video_size_MB', 'video_width'}

#scrape the post urls. # WORKS.
for url in posts:
	try:
		url_dict[url] = list(facebook_scraper.get_posts(post_urls = [url], options=my_fb_options, 
			timeout=fb_timeout, cookies = cookie_path))
	except HTTPError:
		#return none and add url to a list if there was a problem.
		dead_urls.append(url)
		print(url + 'could not be retrieved')
	except KeyboardInterrupt:
		exit()

	for item in stuff_we_dont_need:
		url_dict[url][0].pop(item, None)

print('\n scraping posts successfull, saving... \n ')
#save it to hDD in case of crash etc
filehandler = open(filenames[1], 'wb')  
pickle.dump(fb_data, filehandler)
filehandler.close()

#get pages
for url in pages:
	page_id = re.search('facebook.com/([^/]+)/', url).groups()[0]
	try:
		url_dict[url] = list(facebook_scraper.get_posts(account = page_id, options=my_fb_options,
			timeout =fb_timeout, pages = num_fb_pg, cookies = cookie_path))
	except HTTPError:
		#return none and add url to a list if there was a problem.
		dead_urls.append(url)
		print(url + 'could not be retrieved')
	except KeyboardInterrupt:
		exit()

	for post_dict in url_dict[url]:
		for item in stuff_we_dont_need:
			post_dict.pop(item, None)


print('\n saving dead URLs... \n ')
#save URLs which we couldnt retrieve
with open(filenames[2], "w") as output:
	output.write(str(dead_urls))

print('\n scraping pages successfull, saving... \n ')
#save to pkl
#save it to hDD in case of crash etc
filehandler = open(filenames[3], 'wb')  
pickle.dump(fb_data, filehandler)
filehandler.close()


