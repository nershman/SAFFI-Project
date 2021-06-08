# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-26 17:36:15
# @Last Modified by:   sma
# @Last Modified time: 2021-06-08 18:52:32
"""
TODO: modify this to be a class where you first do 

newthing = metrics_helpers.function(dict, fb=True) <- construct a new object where a certain dict is references/asgned at the construct
then you just do
newthing.add_variable()

((then we can also just tokenize text once, and i guess provide options in initialization for that as well))

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

from lexical_diversity import lex_div as ld

from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
import pycld2 as cld2


################################
# helper functions to class#
################################

def get_counts_from_text_dict(text_dict):
	"""
	Takes a dict  of keys of URLs and value as string.
	"""
	vocab = scr.get_concerns() + scr.get_foods()
	vocab = {substring.lower().strip() for item in vocab for substring in item.split()}
	term_counter = CountVectorizer(vocabulary = vocab, stop_words = 'english')
	return term_counter.fit_transform(text_dict.values()), term_counter

	#TODO: convert the matrix fb_spars into a pandas dataframe 
	#NOTE: From Python 3.6 onwards, the standard dict type maintains insertion order by default.
	#https://stackoverflow.com/questions/1867861/how-to-keep-keys-values-in-same-order-as-declared

def clustering(text_dict):


class indicators:
	"""
	This class takes a dictionary of a certian structure (netmums or facebook results_dict)
	and functions can be called to add different values to it.
	"""
	def __init__(self, results_dict, fb):
		self.results_dict = results_dict
		self.fb = fb
		#TODO generate the strings 

	def get_dict(self, non_ind=True):
		#TODO: if non_ind false then return only indicators
		return self.results_dict

	def get_text_dict(self, remove_links = True):
		if fb:
			text_dict = {key: ' '.join([str(item['text']) for item in value['data']] + \
			[' '.join([str(c['comment_text']) for c in item['comments_full']]) for item in value['data'] if item['comments_full']]
			) for key, value in self.results_dict.items()}
		else:
			text_dict = {key: value['title'] + ' ' + \
			' '.join([str(item['body']) for item in value['posts']]) for key, value in self.results_dict.items()}
		if remove_links:
			regex = r'http\S+'
			textdict = {key: re.sub(regex, '', value) for key, value in textdict.items()}
		return text_dict

	def add_term_counts(self):
		"""
		For each key in a dict, add a new key to it's value (also a dict) named 'term_counts' containing a list thing.
		"""
		textdict = self.get_text_dict()

		fb_spars, term_counter = get_counts_from_text_dict(textdict)
		fb_spars = fb_spars.toarray()
		#add new key to dict with labeled count items 
		for num, key in enumerate(textdict.keys()):
			#dict of 'term':count for each document
			self.results_dict[key]['term_counts'] = {key: fb_spars[num][value] for key, value in term_counter.vocabulary_.items()}
		return
	
	
	def add_url_term_counts(self):
		if fb:
			#get all the fb text
			textdict = {key: ' '.join([str(item['post_text']) for item in value['data']] + \
			[str(item['text']) for item in value['data']] + \
			[' '.join([str(c['comment_text']) for c in item['comments_full']]) for item in value['data'] if item['comments_full']]
			) for key, value in self.results_dict.items()}
			#extract the links from the text using regex
			regex = 'http\S+'
			textdict = {key: ' '.join(re.findall(regex, value)) for key, value in textdict.items()}
	
		else:
			textdict = {key: ' '.join([' '.join([subitem for subitem in item['body_urls']]) for item in value['posts']]) for key, value in self.results_dict.items()}
		#analyze
		fb_spars, term_counter = get_counts_from_text_dict(textdict)
		fb_spars = fb_spars.toarray()
	
		#add new key to dict with labeled count items 
		for num, key in enumerate(textdict.keys()):
			#dict of 'term':count for each document
			self.results_dict[key]['url_term_counts'] = {key: fb_spars[num][value] for key, value in term_counter.vocabulary_.items()}
		return
	
	def add_total_likes(self):
		"""
		for facebook: return likes of a post (not comments)
		for netmums: return total likes of comments in a thread
		"""
		if fb:
			likes = {key: np.sum([post['likes'] for post in value['data']]) for key, value in self.results_dict.items()}
		else:
			likes = {key: np.sum([int(post['likes']) for post in value['posts']]) for key, value in self.results_dict.items()}
	
		#add key to dict
		for key, value in likes.items():
			self.results_dict[key]['total_likes'] = value
	
		return
	
	def add_available_comments(self):
		"""
		ONLY for the facebook count length of comments
		for netmums is returns the same number as get_comment_activity
		"""
		if fb:
			num_c = {key: np.sum([len(post['comments_full']) for post in value['data'] if post['comments_full']]) for key, value in self.results_dict.items()}
		else:
			num_c = {key: len(value['posts']) for key, value in self.results_dict.items()}
	
		#add key to dict
		for key, value in num_c.items():
			self.results_dict[key]['available_comments'] = value
		return
	
	def add_comment_activity(self):
		"""
		returns reported number of comments for FB and for netmums counts avail comments
		"""
		if fb:
			num_c = {key: np.sum([post['comments'] for post in value['data']]) for key, value in self.results_dict.items()}
		else:
			num_c = {key: len(value['posts']) for key, value in self.results_dict.items()}
	
		#add key to dict
		for key, value in num_c.items():
			self.results_dict[key]['comment_activity'] = value
	
		return
	
	
	def add_num_unique_posters(self):
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
			names = {key: {post['username'] for post in thread['posts']} for key, thread in self.results_dict.items()}
	
		#count the size of each set of unique names
		names = {key: len(value) for key, value in names.items()}
	
		#add key to dict
		for key, value in names.items():
			self.results_dict[key]['num_unique_posters'] = value
	
		return
	
	def add_num_urls(self):
		if fb:
			#get all the fb text
			body_urls = {key: ' '.join([str(item['post_text']) for item in value['data']] + \
			[str(item['text']) for item in value['data']] + \
			[' '.join([str(c['comment_text']) for c in item['comments_full']]) for item in value['data'] if item['comments_full']]
			) for key, value in self.results_dict.items()}
			#extract the links from the text using regex
			#regex = '/([\w+]+\:\/\/)?([\w\d-]+\.)*[\w-]+[\.\:]\w+([\/\?\=\&\#\.]?[\w-]+)*\/?/gm'
			regex = 'http\S+'
			body_urls = {key: re.findall(regex, value) for key, value in body_urls.items()}
	
		else:
			body_urls = {key: [url for post in thread['posts'] for url in post['body_urls']] for key, thread in self.results_dict.items()}
	
		#count the size of each set of unique names
		body_urls = {key: len(value) for key, value in body_urls.items()}
	
		#add key to dict
		for key, item in body_urls.items():
		#dict of 'term':count for each document
			self.results_dict[key]['num_urls'] = item
		return
	
	def add_avg_comment_length(self): 
		if not fb: #TODO make this a warning.
			print("add_avg_comment_length is only applicable for facebook. no data added.")
			return

		#get a list of lengths of comments for each post
		length_dict = {key: [len(c['comment_text']) for c in item['comments_full']] \
			for key, value in self.results_dict.items() for item in value['data'] if item['comments_full']}
		length_dict = {key: np.mean(value) for key, value in length_dict.items() if value}
	
		#add key to dict
		for key, item in length_dict.items():
			self.results_dict[key]['avg_comment_length'] = item
		return	
	
	
	def add_avg_post_length(self):
		if fb:
			length_dict = {key: [len(item['text']) for item in value['data'] if item['text']] for key, value in self.results_dict.items()}
		else:
			length_dict = {key: [len(post['body']) for post in thread['posts']] for key, thread in self.results_dict.items()}
	
		#take avg
		length_dict = {key: np.mean(value) for key, value in length_dict.items()}
		#add key to dict
		for key, item in length_dict.items():
			self.results_dict[key]['avg_post_length'] = item
		return	
	
	def add_post_time(self):
		"""
		facebook: returns a datetime object corresponding ot the post
		netmums: returns a list of datetime corresponding to date of each post in a thread.
		"""
		if fb:
			dt_dict = {key: item['time'] for key, value in self.results_dict.items() for item in value['data']}
		else:
			dt_dict = {key: [post['date'] for post in thread['posts']] for key, thread in self.results_dict.items()}
			dt_dict = {key: datetime.strptime(item, '%m/%d/%Y at %I:%M %p') for key, value in dt_dict.items() for item in value}
		#add key to dict
		for key, item in dt_dict.items():
			self.results_dict[key]['post_time'] = item
		return
	
	def add_post_language(self): #facebook only! (we dont need it for netmums)
		"""
		tag  posts language using package cld2
		"""
		#process text to single strings
		textdict = self.get_text_dict()

		for key, value in textdict.items():
			try:
				_, _, details = cld2.detect(value, bestEffort = True)
			except: #if there's a unicode error which throws pycld2.error then we want to remove those unicodes
				printable_str = ''.join(x for x in value if x.isprintable())
				_, _, details = cld2.detect(printable_str, bestEffort = True)
			#dict of 'term':count for each document
			self.results_dict[key]['post_language'] = details[0]
		return
		
	
	#TODO untested
	def add_num_quotes(self): #ONLY netmums
		pass #TODO
	
	def add_lexical_richness(self): #AKA vocabulary diversity
		"""
		We do not use TTR (Type-to-Token Ratio) because it introduces a bias for size of sentences. for each document (URL)
		Source: https://github.com/jennafrens/lexical_diversity "has an inverse ..."
		We use MTLD instead. (we could also use HDD?)
	
		"""
		textdict = self.get_text_dict()

		#generate TTF for each document or something....
	
		diversity = {}
		for key in textdict.keys():
			diversity[key] = ld.mtld(textdict[key].split())
		
		#add new key to dict with labeled count items 
		for key, value in diversity.items():
			#dict of 'term':count for each document
			self.results_dict[key]['lexical_richness'] = value
		return
	
	def add_ttr(self):
		"""
		Implement TTR just to see if its useful.
		"""
		textdict = self.get_text_dict() #TODO: UNTESTED

		#generate TTF for each document or something....
	
		diversity = {}
		for key in textdict.keys():
			diversity[key] = ld.ttr(textdict[key].split())
		
		#add new key to dict with labeled count items 
		for key, value in diversity.items():
			#dict of 'term':count for each document
			self.results_dict[key]['lexical_richness'] = value
		return


	#############################
	## #MORE COMPLICATED METRICS#
	#############################
	
	def add_topic_cluster(self):
		pass

	def add_hazard_product_distance(self):
		pass

	def add_serious_measure(self):
		pass
	
	def add_transitivity_score(self):
		pass	