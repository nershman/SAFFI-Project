# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-04-19 15:22:28
# @Last Modified by:   sma
# @Last Modified time: 2021-04-22 21:33:30
"""
This class builds a list of query URLs and gets the resulting URLs from the search results,
number of results for each query, and possibly the blurb of each result.

Basic search queries are limite to 10 results and 10 pages == 100 total results per query.
Fore more comprehensive results, using POST headers (and other stuff maybe) is needed.
"""

import requests
from bs4 import BeautifulSoup
import time
import re

#PLAN 
#1. create a function to get URLS of searches :)
#2.0 start to build function that will load the URLs x
	#2.1 see if I end up getting requests rejected x 
	#2.2 find good time interval x 
#3. make the function get next n pages of search as well
	#3.1 for many results, the search will just duplicate prev pages when u go past ten
	#we can just grab by url the pages from 1 to ten but also first check the first page see how mnay 
	#pages of results there are. 
#4. make the function scrape those DISCUSSION-URLS
#5. make NEW function which scrapes text (and name?) from those pages.



def build_search_urls(lst):
	"""
	This function uses netmums basic URL search which is faster
	but has less functionality.
	Return list of URLS from a list of search terms.
	"""
	if not lst:
		raise TypeError('list has no values')
	return (['https://www.netmums.com/search/chat/' + terms for terms in lst])

def get_next_pages(urlstring, l = 10):
	"""
	Returns a lit of string URL with /page:[num] appended for num in 1 to l.

	l: number of pages of results (btwn 2 and 10)

	If applying in lambda or list, pass l = None and the function will return None.
	"""
	if l is None:
		num_pages = None
	elif l > 10 or l < 0:
		raise InputError('l must be between 1 and 10 inclusive')
	elif bool(re.search('page:[0-9]',urlstring)):
		raise Error('do not use URLs with a page number in them')

	else:
		num_pages = [urlstring + '/page:' + str(digit) for digit in range(2,l + 1)]
	return num_pages

def extract_results(soup, blurbs = False, titles = False):
	"""
	TODO
	extract links from a page of netmums basic search results

	right now titles and blurbs options do nothing lol
	"""
	
	found = soup.find_all('h3', {'class', 'card__title'})
	if found is not None:
		#TODO check if blurbs or title wanted
		links = [link.find('a').get('href') for link in found]

		if titles:
			title_text = [link.find('a').text for link in found]

		if blurbs:
			blurb_text = [link.find_next_sibling(
				'p', {'class':'card__text'}).text for link in found]

		#create a list of dicts to return and save it to found
		if titles and blurbs:
			found = [{'link':tup[0],
					  'title':tup[1],
					  'blurb':tup[2]} for tup in list(zip(links, title_text, blurb_text))]
		elif titles:
			found = [{'link':tup[0],
					  'title':tup[1]} for tup in list(zip(links,title_text))]
		elif blurbs:
			found = [{'link':tup[0],
					  'blurb':tup[1]} for tup in list(zip(links,blurb_text))]
		else:
			found = [{'link':l} for l in links]
	return found

def count_results(soup):
	"""
	Returns the number of results from a search page.
	"""
	found = soup.find('p', {'class':'search-results__count'})

	if found is not None:
		found = int(re.search('\d+\s', found.text).group(0))

		return found

def num_pages(soup):
	"""
	Returns the number of pages of a basic search query
	"""
	found = soup.find('p', {'class':'pagination__count'})
	if found is not None:
		found = int(re.search('\d+$', found.text).group(0))

	return found

def get_res_from_url(url, rate=0.05, blurbs = False, titles = False):
	"""
	'url': a single string corresponding to the URL of the
		   first page of a netmums basic search query

	Returns up to ten pages of results from a basic search query URL,
	in the form of a list of dicts with key 'link' and optional attributes
	'blurbs' and 'titles'

	default rate limit at 0.05 seconds per request

	"""
	#get the html of a search query
	html = requests.get(url).text

	#build list of URLS
	next_pages = get_next_pages(url, num_pages(BeautifulSoup(html, 'html.parser')))

	#save all the htmls to a list
	if next_pages is not None:
		all_htmls = [html] + [requests.get(u).text for u in next_pages]

	else:
		all_htmls = [html]

	#get the results for each URL
	#list of lists of dicts, one list of dicts per page.
	results = [extract_results(BeautifulSoup(h, 'html.parser'),
								  titles=titles,
								  blurbs=blurbs
								  ) for h in all_htmls]

	#flatten the list of lists 
	results = [item for dictlist in results for item in dictlist]

	#return url, number of results, and all the htmls, and datatype that ocntains blurb and user and title.
	return results


#TODO test it.
def get_res_from_list(urllist, rate = 0.01, urlrate=0.05, blurbs = False, titles = False):
	"""
	takes list of urls and performs get_res_from_url on them. 
	Returns a dict of key=queryURL value = list of results
	"""
	return dict(zip(urllist,
					[get_res_from_url(url, rate=rate, blurbs = blurbs, titles = titles
						) for url in urllist if time.sleep(rate) is None]
					))


#NOTE TO DO (MAYBE): 
# the queries only gather a limited number (100 = 10 res x 10 pages) of results
# so for now I would like to just save the number of results returned as well
# which could make it easier to figure out later.
# PHP quereis seem to be way slower (https://www.netmums.com/coffeehouse/search_result.php?config=work-692.inc)


############## CRAP