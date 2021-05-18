# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-11 13:00:46
# @Last Modified by:   sma
# @Last Modified time: 2021-05-16 19:55:44

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
from search_engines import http_client as search_http_client
import facebook_scraper
from netmums import scrapehelpers as scr #requires making parent path visible
from facebook import facebook_helper as facehelp
import pickle 
import time
import re
from random import uniform as random_uniform
from urllib.error import HTTPError

################
##  set up	##
################

#in case we need to re-run to capture more links, we can easily update the file names from here.
filenames = ['fb_search_results.pkl',
'fb_safety.pkl',
'deadURLs.txt',
'fb_data_searchscrape.pkl',
'google_search_results_temp.pkl']

########################################
## import and convert dict to urldict ##
########################################
with open('manual_search_resdict.pkl', 'rb') as f:
	results_dict = pickle.load(f)



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


######################################
## use facebook scraper to get info ##
######################################

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
	# match a FB URl that includes title id (a-string-of-text-like-this)
	url_parts = re.search('facebook.com/([^/]+)/([^/]+)/([^/]+)/([^/]+)', url)
	if not url_parts: #or without the text part in it
		url_parts = re.search('facebook.com/([^/]+)/([^/]+)/([^/]+)', url).groups()
	else:
		url_parts = url_parts.groups()

	#build string in format $page_id/posts/$post_id
	scrape_id = url_parts[0]+'/'+url_parts[1]+'/'+url_parts[-1]

	try:
		url_dict[url] = list(facebook_scraper.get_posts(post_urls = [scrape_id], options=my_fb_options, 
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
pickle.dump(url_dict, filehandler)
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
	except AttributeError:
		url_dict[url] = ['error']

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
pickle.dump(url_dict, filehandler)
filehandler.close()


