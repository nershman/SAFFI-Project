# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-26 17:36:15
# @Last Modified by:   sma
# @Last Modified time: 2021-06-01 20:10:43
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
with open('fb_merged_cleaned_flat.pkl', 'rb') as f:
    fb_search = pk.load(f)
#####################
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
import pycld2 as cld2
#TODO: convert the matrix fb_spars into a pandas dataframe 
#NOTE: From Python 3.6 onwards, the standard dict type maintains insertion order by default.
#https://stackoverflow.com/questions/1867861/how-to-keep-keys-values-in-same-order-as-declared
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

def add_term_counts(results_dict, fb=False):
	"""
	For each key in a dict, add a new key to it's value (also a dict) named 'term_counts' containing a list thing.
	"""
	if fb:
		#extract text from dict and build a dict where values are strings of text
		textdict = {key: ' '.join([str(item['text']) for item in value['data']] + \
		[' '.join([str(c['comment_text']) for c in item['comments_full']]) for item in value['data'] if item['comments_full']]
		) for key, value in results_dict.items()}
	else:
		textdict = {key: value['title'] + ' ' + \
		' '.join([str(item['body']) for item in value['posts']]) for key, value in results_dict.items()}
	#remove all links from the text
	regex = r'http\S+'
	textdict = {key: re.sub(regex, '', value) for key, value in textdict.items()}
				#FIXME: not sure if shoul dhav ethe join or not (i deleted in this one.)
	#analyze
	fb_spars, term_counter = get_counts_from_text_dict(textdict)
	fb_spars = fb_spars.toarray()
	#add new key to dict with labeled count items 
	for num, key in enumerate(textdict.keys()):
		#dict of 'term':count for each document
		results_dict[key]['term_counts'] = {key: fb_spars[num][value] for key, value in term_counter.vocabulary_.items()}
	return


def add_url_term_counts(results_dict, fb=False):
	if fb:
		#get all the fb text
		textdict = {key: ' '.join([str(item['post_text']) for item in value['data']] + \
		[str(item['text']) for item in value['data']] + \
		[' '.join([str(c['comment_text']) for c in item['comments_full']]) for item in value['data'] if item['comments_full']]
		) for key, value in results_dict.items()}
		#extract the links from the text using regex
		regex = 'http\S+'
		textdict = {key: ' '.join(re.findall(regex, value)) for key, value in textdict.items()}

	else:
		textdict = {key: ' '.join([' '.join([subitem for subitem in item['body_urls']]) for item in value['posts']]) for key, value in results_dict.items()}
	#analyze
	fb_spars, term_counter = get_counts_from_text_dict(textdict)
	fb_spars = fb_spars.toarray()

	#add new key to dict with labeled count items 
	for num, key in enumerate(textdict.keys()):
		#dict of 'term':count for each document
		results_dict[key]['url_term_counts'] = {key: fb_spars[num][value] for key, value in term_counter.vocabulary_.items()}
	return

def add_total_likes(results_dict, fb=False):
	"""
	for facebook: return likes of a post (not comments)
	for netmums: return total likes of comments in a thread
	"""
	if fb:
		likes = {key: np.sum([post['likes'] for post in value['data']]) for key, value in results_dict.items()}
	else:
		likes = {key: np.sum([int(post['likes']) for post in value['posts']]) for key, value in results_dict.items()}

	#add key to dict
	for key, value in likes.items():
		results_dict[key]['total_likes'] = value

	return

def add_available_comments(results_dict, fb=False):
	"""
	ONLY for the facebook count length of comments
	for netmums is returns the same number as get_comment_activity
	"""
	if fb:
		num_c = {key: np.sum([len(post['comments_full']) for post in value['data'] if post['comments_full']]) for key, value in results_dict.items()}
	else:
		num_c = {key: len(value['posts']) for key, value in results_dict.items()}

	#add key to dict
	for key, value in num_c.items():
		results_dict[key]['available_comments'] = value
	return

def add_comment_activity(results_dict, fb=False):
	"""
	returns reported number of comments for FB and for netmums counts avail comments
	"""
	if fb:
		num_c = {key: np.sum([post['comments'] for post in value['data']]) for key, value in results_dict.items()}
	else:
		num_c = {key: len(value['posts']) for key, value in results_dict.items()}

	#add key to dict
	for key, value in num_c.items():
		results_dict[key]['available_comments'] = value

	return


def add_num_unique_posters(results_dict, fb=False):
	names = {}
	if fb:
		for key, value in fb_search.items():
			names[key] = set()
			for item in value['data']:
				if item['username']:
					names[key].add(item['username'])
				if item['shared_username']:
					names[key].add(item['shared_username'])
				if item['comments_full']:
					for c in item['comments_full']:
						if c['commenter_name']:
							names[key].add(c['commenter_name'])
	else:
		names = {key: {post['username'] for post in thread['posts']} for key, thread in results_dict.items()}

	#count the size of each set of unique names
	names = {key: len(value) for key, value in names.items()}

	#add key to dict
	for key, value in names.items():
		results_dict[key]['num_unique_posters'] = value

	return

def add_num_urls(results_dict, fb=False):
	if fb:
		#get all the fb text
		body_urls = {key: ' '.join([str(item['post_text']) for item in value['data']] + \
		[str(item['text']) for item in value['data']] + \
		[' '.join([str(c['comment_text']) for c in item['comments_full']]) for item in value['data'] if item['comments_full']]
		) for key, value in results_dict.items()}
		#extract the links from the text using regex
		#regex = '/([\w+]+\:\/\/)?([\w\d-]+\.)*[\w-]+[\.\:]\w+([\/\?\=\&\#\.]?[\w-]+)*\/?/gm'
		regex = 'http\S+'
		body_urls = {key: re.findall(regex, value) for key, value in body_urls.items()}

	else:
		body_urls = {key: [url for post in thread['posts'] for url in post['body_urls']] for key, thread in results_dict.items()}

	#count the size of each set of unique names
	body_urls = {key: len(value) for key, value in body_urls.items()}

	#add key to dict
	for key, item in body_urls.items():
	#dict of 'term':count for each document
		results_dict[key]['num_urls'] = item
	return

def add_avg_comment_length(results_dict): 
	#facebook ONLY.
	#fb = True
	#get a list of lengths of comments for each post
	length_dict = {key: [len(c['comment_text']) for c in item['comments_full']] \
		for key, value in results_dict.items() for item in value['data'] if item['comments_full']}
	length_dict = {key: np.mean(value) for key, value in length_dict.items() if value}

	#add key to dict
	for key, item in length_dict.items():
		results_dict[key]['avg_comment_length'] = item
	return	


def add_avg_post_length(results_dict, fb=False):
	if fb:
		length_dict = {key: [len(item['text']) for item in value['data'] if item['text']] for key, value in results_dict.items()}
	else:
		length_dict = {key: [len(post['body']) for post in thread['posts']] for key, thread in results_dict.items()}

	#take avg
	length_dict = {key: np.mean(value) for key, value in length_dict.items()}
	#add key to dict
	for key, item in length_dict.items():
		results_dict[key]['avg_post_length'] = item
	return	

def add_num_comments(results_dict, fb=False):
	if fb:
		num_dict = {key: len(item['comments_full']) for key, value in results_dict.items() for item in value['data'] if item['comments_full']}
	else:
		num_dict = {key: len(thread['posts']) for key, thread in results_dict.items()}
	#add key to dict
	for key, item in num_dict.items():
		results_dict[key]['num_posts'] = item
	return

def add_post_time(results_dict, fb=False):
	"""
	facebook: returns a datetime object corresponding ot the post
	netmums: returns a list of datetime corresponding to date of each post in a thread.
	"""
	if fb:
		dt_dict = {key: item['time'] for key, value in results_dict.items() for item in value['data']}
	else:
		dt_dict = {key: [post['date'] for post in thread['posts']] for key, thread in results_dict.items()}
		dt_dict = {key: datetime.strptime(item, '%m/%d/%Y at %I:%M %p') for key, value in dt_dict.items() for item in value}
	#add key to dict
	for key, item in dt_dict.items():
		results_dict[key]['post_time'] = item
	return

#FIXME: it hangs forever(?) or its hells slow idk.
def add_post_language(results_dict, fb = True): #facebook only! (we dont need it for netmums)
	"""
	tag  posts language using package cld2
	"""
	#process text to single strings
	if fb:
		textdict = {key: ' '.join([str(item['text']) for item in value['data']] + \
		[' '.join([str(c['comment_text']) for c in item['comments_full']]) for item in value['data'] if item['comments_full']]
		) for key, value in results_dict.items()}
	else: #netmums is kind of superfluos since we know the entire forum is in english
		textdict = {key: value['title'] + ' ' + \
		' '.join([str(item['body']) for item in value['posts']]) for key, value in results_dict.items()}

	#remove all links from the text
	regex = r'http\S+'
	textdict = {key: re.sub(regex, '', value) for key, value in textdict.items()}

	for key, value in textdict.items():
		try:
			_, _, details = cld2.detect(value, bestEffort = True)
		except: #if there's a unicode error which throws pycld2.error then we want to remove those unicodes
			printable_str = ''.join(x for x in value if x.isprintable())
			_, _, details = cld2.detect(printable_str, bestEffort = True)
		#dict of 'term':count for each document
		results_dict[key]['post_language'] = details[0]
	return
	


def num_quotes(results_dict): #ONLY netmums
	pass #TODO