# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-28 12:34:21
# @Last Modified by:   sma
# @Last Modified time: 2021-05-30 10:47:56
import os #DEBUG
os.chdir('/Users/sma/Documents/INRAE internship/scrape-git') #DEBUG
import pickle as pk
import re
import sys
sys.path.append('../')
from facebook import facebook_helper as facehelp


with open('facebook/fb_data_searchscrape.pkl', 'rb') as f:
    fb_search = pk.load(f)
with open('facebook/specific_fb_pages.pkl', 'rb') as f:
    pages = pk.load(f)
with open('facebook/specific_fb_groups.pkl', 'rb') as f:
    groups = pk.load(f)

with open('facebook/manual_search_resdict.pkl', 'rb') as f:
    fb_resultdict = pk.load(f)


#ADD THE QUERIES TO fb_searchd dict.
#i made a mistake when creating the facebook data so I need to fix it.
fb_urldict = facehelp.resultsdict_to_urldict(fb_resultdict)
#we need to add the query backin for each.
for key in fb_urldict.keys():
    #both dicts should have the exact same keys
    if key in fb_search.keys() and type(fb_search[key]) is list:
        fb_search[key] = {'query':fb_urldict[key]['query'], 'data':fb_search[key]}

#REFORMAT THE OTHER TWO TO HAVE THE SAME DATA STRUCTURE
for key in pages:
	pages[key] = {'query': {'specific_page'}, 'data':pages[key]}

for key in groups:
	groups[key] = {'query': {'specific_group'}, 'data':groups[key]}

#JOIN THE THREE TOGETHER INTO ONE.
#if I want to select on the searches later, I can just limit to keys with http in it :)
fb_joined = {key:value for item in [fb_search] for key, value in item.items()}



#REMOVE POSTS WHICH ARE EMPTY
delete_these = {}
for key in fb_joined.keys():
	for ind, item in enumerate(fb_joined[key]['data']):
		if item['post_id'] is None:
			if key in delete_these:
				#the key exists so there is already a set.
				delete_these[key].add(ind)
			elif len(fb_joined[key]['data']) <= 1:
				delete_these[key] = {'ENTIRE'}
			else:
				delete_these[key] = {ind}

for key, post_list in delete_these.items():
	if  'ENTIRE' in post_list:
		fb_joined.pop(key)
	else:
		for ind in post_list:
			value.remove(ind)

#FLATTEN THE PAGE ONES SO WE JSUT HAVE SINGLE POST OBJECTS.
search_posts = [key for key in fb_joined.keys() if re.search('posts/[^?]+',key)]
search_pages = [key for key in fb_joined.keys() if not re.search('posts/[^?]+',key)]

fb_flat = {}
for key in search_posts:
	fb_flat[key] = fb_joined[key]
for key in search_pages:
	for item in fb_joined[key]['data']:
		fb_flat[item['post_url']] = [item]

#save to a new thing.
with open('fb_merged_cleaned_flat.pkl', 'wb') as f:
	pk.dump(fb_joined,f)