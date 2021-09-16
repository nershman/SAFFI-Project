# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-16 14:41:57
# @Last Modified by:   sma
# @Last Modified time: 2021-05-18 11:05:41

import sys
sys.path.append('../') #make parent path visible so we can import modules from other folders.

from search_engines import Duckduckgo
import facebook_scraper
from netmums import scrapehelpers as scr #requires making parent path visible
from facebook import facebook_helper as facehelp
import pickle 
import time
import re
from random import uniform as random_uniform
from urllib.error import HTTPError

pages = [
'LondonMums',
'573288336382252',
'200655304061',
'28069069250',
'431263726919454'
]

#new one was 50 x 200
my_fb_options = {"posts_per_page": 20, "comments":True, "reactors": False, "reactions":True}
cookie_path = '/Users/sma/Documents/INRAE internship/scrape_files_confidential/fb_cookies.txt'
num_fb_pg = 30
fb_timeout = 45
stuff_we_dont_need = {'video', 'image_lowquality', 'images', 'image', 
						'is_live', 'shared_post_id', 'shared_post_url', 
						'video', 'video_thumbnail', 'video_duration_seconds', 
						'video_height', 'video_id', 'video_quality', 
						'video_size_MB', 'video_width'}


url_dict = {}
for url in pages:
	page_id = url
	try:
		url_dict[url] = list(facebook_scraper.get_posts(group = page_id, options=my_fb_options,
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


#save to pkl
#save it to hDD in case of crash etc
filehandler = open('specific_fb_groups.pkl', 'wb')  
pickle.dump(url_dict, filehandler)
filehandler.close()