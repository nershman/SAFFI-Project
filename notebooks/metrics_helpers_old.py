# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-26 17:36:15
# @Last Modified by:   sma
# @Last Modified time: 2021-05-30 11:03:04
"""
Goal: build a set of functions to create our metrics.
In the end, I think we will use a dataframe where each obs is a URL corresponding to 
a key in one of our dicts.

#TODO: make sure we differentiate facebook POSTS from facebook pages
#TODO: we probably should separate out post somehow.
	- maybe in the dataframe i can put the index of the specific post as well 


UPDATE: this file is unused.

helper functions using skleaern to vectorize and count terms in our datatypes.

https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
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
import time
import re
from random import uniform as random_uniform
from urllib.error import HTTPError
import sys
import pickle as pk
import numpy as np
from pprint import pprint 

#open the facebook data
with open('fb_data_searchscrape.pkl', 'rb') as f:
    fb_search = pk.load(f)

from sklearn.feature_extraction.text import CountVectorizer

#TODO: convert the matrix fb_spars into a pandas dataframe 

################################
# helper functions to main ones#
################################

def get_counts_from_text_dict(text_dict):
	vocab = scr.get_concerns() + scr.get_foods()
	vocab = {substring.lower().strip() for item in vocab for substring in item.split()}
	term_counter = CountVectorizer(vocabulary = vocab, stop_words = 'english')
	return term_counter.fit_transform(text_dict.values()), term_counter


#######################
# FUNCTIONS I WILL USE#
#######################
"""
"""

#TODO: UNTESTED
def get_term_counts(result_dict, fb=False):
	"""
	For each key in a dict, add a new key to it's value (also a dict) named 'term_counts' containing a list thing.
	"""
	if fb:
		#extract text from dict and build a dict where values are strings of text
		textdict = {key: ' '.join([str(item['post_text']) for item in value['data']] + \
		[str(item['text']) for item in value['data']] + \
		[' '.join([str(c['comment_text']) for c in item['comments_full']]) for item in value['data'] if item['comments_full']]
		) for key, value in result_dict.items()}
	else:
		textdict = {key: value['title'] + ' ' + \
		' '.join([str(item['body']) for item in value['posts']]) for key, value in netmums.items()}

	#remove all links from the text
	regex = '/([\w+]+\:\/\/)?([\w\d-]+\.)*[\w-]+[\.\:]\w+([\/\?\=\&\#\.]?[\w-]+)*\/?/gm'
	textdict = {key: ' '.join(re.sub(regex, '', value)) for key, value in textdict.items()}
	
	#analyze
	fb_spars, term_counter = get_counts_from_text_dict(textdict)
	fb_spars = fb_spars.toarray()

	#add new key to dict with labeled count items 
	for num, key in enumerate(textdict.keys()):
		#dict of 'term':count for each document
		result_dict[key]['term_counts'] = {key: fb_spars[num][value] for key, value in term_counter.vocabulary_.items()}
	return

#TODO: UNTESTED
def get_url_term_counts(result_dict, fb=False):
	if fb:
		#get all the fb text
		textdict = {key: ' '.join([str(item['post_text']) for item in value['data']] + \
		[str(item['text']) for item in value['data']] + \
		[' '.join([str(c['comment_text']) for c in item['comments_full']]) for item in value['data'] if item['comments_full']]
		) for key, value in result_dict.items()}
		#extract the links from the text using regex
		regex = '/([\w+]+\:\/\/)?([\w\d-]+\.)*[\w-]+[\.\:]\w+([\/\?\=\&\#\.]?[\w-]+)*\/?/gm'
		textdict = {key: ' '.join(re.findall(regex, value)) for key, value in textdict.items()}

	else:
		textdict = {key: ' '.join([' '.join([subitem for subitem in item['body_urls']]) for item in value['posts']]) for key, value in netmums.items()}
	#analyze
	fb_spars, term_counter = get_counts_from_text_dict(textdict)
	fb_spars = fb_spars.toarray()

	#add new key to dict with labeled count items 
	for num, key in enumerate(textdict.keys()):
		#dict of 'term':count for each document
		result_dict[key]['url_term_counts'] = {key: fb_spars[num][value] for key, value in term_counter.vocabulary_.items()}
	return

#TODO 
def get_total_likes(results_dict, fb=False):
	"""
	for facebook: return likes of a post (not comments)
	for netmums: return total likes of comments in a thread
	"""
	if fb:
		likes = {key: np.sum([post['likes'] for post in value['data']]) for key, value in results_dict.items()}
	else:
		likes = {key: np.sum([post['likes'] for post in value['posts']]) for key, value in results_dict.items()}

	for key, value in likes.items():
		result_dict[key]['total_likes'] = value

def get_available_comments(results_dict, fb=False):
	"""
	ONLY for the facebook count length of comments
	for netmums is returns the same number as get_comment_activity
	"""
	if fb:
		num_c = {key: np.sum([len(post['comments_full']) for post in value['data']]) for key, value in results_dict.items()}
	else:
		num_c = {key: len(value['posts']) for key, value in results_dict.items()}

	for key, value in num_c.items():
		result_dict[key]['available_comments'] = value

def get_comment_activity(results_dict, fb=False):
	"""
	returns reported number of comments for FB and for netmums counts avail comments
	"""
	if fb:
		num_c = {key: np.sum([post['comments'] for post in value['data']]) for key, value in results_dict.items()}
	else:
		num_c = {key: len(value['posts']) for key, value in results_dict.items()}

	for key, value in num_c.items():
		result_dict[key]['available_comments'] = value

def get_unique_posters(results_dict, fb=False):
	pass #TODO

def get_num_urls(results_dict, fb=False):
	if fb:
		#get all the fb text
		textdict = {key: ' '.join([str(item['post_text']) for item in value['data']] + \
		[str(item['text']) for item in value['data']] + \
		[' '.join([str(c['comment_text']) for c in item['comments_full']]) for item in value['data'] if item['comments_full']]
		) for key, value in result_dict.items()}
		#extract the links from the text using regex
		regex = '/([\w+]+\:\/\/)?([\w\d-]+\.)*[\w-]+[\.\:]\w+([\/\?\=\&\#\.]?[\w-]+)*\/?/gm'
		textdict = {key: len(re.findall(regex, value)) for key, value in textdict.items()}

	else:
		#TODO netmums

	for key, item in textdict.item():
	#dict of 'term':count for each document
	result_dict[key]['num_urls'] = item
	return

def get_avg_length(results_dict, fb=False):
	pass

def get_num_posts(results_dict, fb=False):
	pass

def get_post_time(results_dict, fb, = False):
	pass