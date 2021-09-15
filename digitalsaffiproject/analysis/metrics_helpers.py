# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-26 17:36:15
# @Last Modified by:   sma
# @Last Modified time: 2021-09-15 16:02:47
"""
helper functions using skleaern to vectorize and count terms in our datatypes.

https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
"""

#import os #DEBUG
#os.chdir('/Users/sma/Documents/INRAE internship/scrape-git/facebook') #DEBUG

import sys
sys.path.append('../')
sys.path.append('../scraping/') #make parent path visible so we can import modules from other folders.

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


#CLUSTERING
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation

import numpy as np

#MINIMUM WORD DISTANCE
from itertools import product


#NOTE: From Python 3.6 onwards, the standard dict type maintains insertion order by default.
#https://stackoverflow.com/questions/1867861/how-to-keep-keys-values-in-same-order-as-declared
################################
# helper functions to class#
################################

# functions for word distance

def get_simple_distance(doc, product_list, hazard_list, type='min'):
	"""
	Document & product are a list of words.

	Returns a distance measure for the selected words in a document.
	"""
	#get relevant terms in the document
	products_intersect = set(product_list).intersection(doc)
	hazards_intersect = set(hazard_list).intersection(doc)

	#get indexes of relevant terms in document
	prod_ind = {doc.index(term) for term in products_intersect}
	haz_ind = {doc.index(term) for term in hazards_intersect}

	#mean distance: find sequences of product-hazard pairs and the distance
	#find the first p/h, then find the second p/h and match it and calc distance.
	#continue to do this for each item.

	#if a term doesn't appear, return NA.
	if not (prod_ind and haz_ind):
		return None

	elif type == 'min':
		vals = min(product(prod_ind, haz_ind), key=lambda t: abs(t[0]-t[1]))
		return abs(vals[0]-vals[1])

	elif type == 'mean':
		pass #TODO

#other things

def get_counts_from_text_dict(text_dict):
	"""
	Takes a dict  of keys of URLs and value as string.
	"""
	vocab = scr.get_concerns() + scr.get_foods()
	vocab = {substring.lower().strip() for item in vocab for substring in item.split()} #this is suffiient for compatibility w countvectorizer.
	term_counter = CountVectorizer(vocabulary = vocab, stop_words = 'english')
	return term_counter.fit_transform(text_dict.values()), term_counter

	#NOTE: From Python 3.6 onwards, the standard dict type maintains insertion order by default.
	#https://stackoverflow.com/questions/1867861/how-to-keep-keys-values-in-same-order-as-declared

#TODO unfinished. theres version done inside a notebook tho.
def clustering(text_dict, chosen_k = 10, n_features = 1000): #FIXME not defined.
	"""
	chosen_k: int
	"""
	text_list = list(text_dict.values())
	keys = list(text_dict.keys())
	#fit and transform data
	vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,min_df=2, 
                             stop_words='english',use_idf=True)
	X = vectorizer.fit_transform(text_list)

	km = MiniBatchKMeans(n_clusters=true_k)
	km.fit(X)
	#TODO: return the fitted values (which cluster each item belongs to)
	return custer_dict

def LDA(): #TODO

	tf_vectorizer = CountVectorizer(max_df=0.5, min_df=2, max_features=n_features, 
                                stop_words='english')
	tf = tf_vectorizer.fit_transform(documents)
	tf_feature_names = tf_vectorizer.get_feature_names()
	lda = LatentDirichletAllocation(n_components=n_topics, 
                                learning_method="batch").fit(tf)
	lda_W = lda.transform(tf)
	lda_H = lda.components_
	return 

class indicators:
	"""
	This class takes a dictionary of a certian structure (netmums or facebook results_dict)
	and functions can be called to add different values to it.
	"""
	def __init__(self, results_dict, fb):
		self.results_dict = results_dict
		self.fb = fb
		self.text_dict = self.get_text_dict()

	def get_dict(self, non_ind=True):
		#TODO: if non_ind false then return only indicators
		return self.results_dict

	def get_text_dict(self, remove_links = True):
		if self.fb:
			text_dict = {key: ' '.join([str(item['text']) for item in value['data']] + \
			[' '.join([str(c['comment_text']) for c in item['comments_full']]) for item in value['data'] if item['comments_full']]
			) for key, value in self.results_dict.items()}
		else:
			text_dict = {key: value['title'] + ' ' + \
			' '.join([str(item['body']) for item in value['posts']]) for key, value in self.results_dict.items()}
		if remove_links:
			regex = r'http\S+'
			text_dict = {key: re.sub(regex, '', value) for key, value in text_dict.items()}
		return text_dict

	def get_posts_dict(self, remove_links = False):
		"""
		Returns dict where key is URL post number pair so posts are easier to iterate over. Discards thread information which is not held in post-list object.
		"""
		if self.fb:
			pass #TODO
		else:
			posts_dict = {(key,n): item for key, value in self.results_dict.items() for n, item in enumerate(value['posts'])}
			#add the title to the first post in each of them
			for url in self.results_dict.keys():
				posts_dict[(url, 0)]['body'] = self.results_dict[url]['title'] + posts_dict[(url, 0)]['body']
		#if remove_links: #TODO
		#	regex = r'http\S+'
		#	posts_dict = {key: re.sub(regex, '', value) for key, value in posts_dict.items()}
		return posts_dict


	def add_term_counts(self):
		"""
		For each key in a dict, add a new key to it's value (also a dict) named 'term_counts' containing a list thing.
		"""
		textdict = self.text_dict

		fb_spars, term_counter = get_counts_from_text_dict(textdict)
		fb_spars = fb_spars.toarray()
		#add new key to dict with labeled count items 
		for num, key in enumerate(textdict.keys()):
			#dict of 'term':count for each document
			self.results_dict[key]['term_counts'] = {key: fb_spars[num][value] for key, value in term_counter.vocabulary_.items()}
		return
	
	
	def add_url_term_counts(self):
		if self.fb:
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
		if self.fb:
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
		if self.fb:
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
		if self.fb:
			num_c = {key: np.sum([post['comments'] for post in value['data']]) for key, value in self.results_dict.items()}
		else:
			num_c = {key: len(value['posts']) for key, value in self.results_dict.items()}
	
		#add key to dict
		for key, value in num_c.items():
			self.results_dict[key]['comment_activity'] = value
	
		return
	
	def add_num_unique_posters(self):
		names = {}
		if self.fb:
			for key, value in self.results_dict.items():
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
		if self.fb:
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
		if not self.fb: #TODO make this a warning.
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
		if self.fb:
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
		if self.fb:
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
		textdict = self.text_dict

		for key, value in textdict.items():
			try:
				_, _, details = cld2.detect(value, bestEffort = True)
			except: #if there's a unicode error which throws pycld2.error then we want to remove those unicodes
				printable_str = ''.join(x for x in value if x.isprintable())
				_, _, details = cld2.detect(printable_str, bestEffort = True)
			#dict of 'term':count for each document
			self.results_dict[key]['post_language'] = details[0]
		return
	
	def add_newline_count(self): #useful for facebook data.
		"""
		motivation posts on facebook with a lot of \n are usually selling stuff.
		"""
		textdict = self.text_dict

		for key, value in textdict.items():
			self.results_dict[key]['newline_count'] = len(re.findall('\n', textdict[key]))
	
	#TODO untested
	def add_num_quotes(self): #ONLY netmums
		pass #TODO
	
	def add_lexical_richness(self): #AKA vocabulary diversity
		"""
		We do not use TTR (Type-to-Token Ratio) because it introduces a bias for size of sentences. for each document (URL)
		Source: https://github.com/jennafrens/lexical_diversity "has an inverse ..."
		We use MTLD instead. (we could also use HDD?)
	
		"""
		textdict = self.text_dict

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
		textdict = self.text_dict

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
	
	def add_term_distance_simple(self, product_list = None, hazard_list = None):
		"""
		Return mean distance between products and hazards
		(note: minimum if mean is too much to implememnt...)
		"""
		if not product_list:
			product_list = {substring.lower().strip() for item in scr.get_foods() for substring in item.split()}
			product_list.remove('for')
			product_list.remove('baby')

		if not hazard_list:
			hazard_list = {substring.lower().strip() for item in scr.get_concerns() for substring in item.split()}
			hazard_list.remove('food')
			hazard_list.remove('authority')
			hazard_list.remove('vet')
			hazard_list.remove('aromatic')
			hazard_list.remove('european')
			hazard_list.remove('a')
			hazard_list.remove('safety')
			hazard_list.remove('sweeteners') #not really sure about this one.
			hazard_list.remove('modified')

		#we make a document into a list, and then count the number of words between e.g "heavy metals" and "baby food"
		textdict = {key: value.split() for key, value in self.text_dict.items()}

		#run fn here
		distance_dict = {key: get_simple_distance(value, product_list, hazard_list, type='min') for \
							key, value in textdict.items()}

		for key, value in distance_dict.items():
			self.results_dict[key]['term_distance_simple'] = value

		return


	def add_topic_cluster(self):
		#textdict = self.text_dict
		pass
		


	def add_hazard_product_distance(self):
		pass

	def add_serious_measure(self):
		pass
	
	def add_transitivity_score(self):
		pass	


## SIMPLE THINGS

	def add_subforum(): #netmums
		pass

	def add_subsubforum(): #netmum
		pass

	def add_num_quotes_with_author():
		pass

	def add_num_quotes_without_author():
		pass