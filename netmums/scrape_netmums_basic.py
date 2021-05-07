# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-04-19 15:22:28
# @Last Modified by:   sma
# @Last Modified time: 2021-05-07 15:42:59
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
import logging
import contextlib
from http.client import HTTPConnection
from requests.packages.urllib3.util.retry import Retry
## SETUP ##
#setup requests session.
s = requests.Session()
retries=Retry(total=7, backoff_factor=1, status_forcelist=[ 502, 503, 504 ])
a = requests.adapters.HTTPAdapter(max_retries=retries)
s.mount('http://', a)
s.mount('https://', a)
tout = 90

def debug_requests_on():
    '''Switches on logging of the requests module.'''
    HTTPConnection.debuglevel = 1

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

def debug_requests_off():
    '''Switches off logging of the requests module, might be some side-effects'''
    HTTPConnection.debuglevel = 0

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.handlers = []
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.WARNING)
    requests_log.propagate = False


#NOTE TO DO (MAYBE): 
# the queries only gather a limited number (100 = 10 res x 10 pages) of results
# so for now I would like to just save the number of results returned as well
# which could make it easier to figure out later.
# PHP quereis seem to be way slower (https://www.netmums.com/coffeehouse/search_result.php?config=work-692.inc)

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
	Returns a list of string URL with /page:[num] appended for num in 1 to l.

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
	extract links from a page of netmums basic search results

	right now titles and blurbs options do nothing lol
	"""
	
	found = soup.find_all('h3', {'class': 'card__title'})
	if found is not None:
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
	html = s.get(url, timeout=tout).text

	#build list of URLS
	next_pages = get_next_pages(url, num_pages(BeautifulSoup(html, 'html.parser')))

	#save all the htmls to a list
	if next_pages is not None:
		all_htmls = [html] + [s.get(u, timeout=tout).text for u in next_pages]

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
	print(url)#DEBUG
	#return url, number of results, and all the htmls, and datatype that ocntains blurb and user and title.
	return results


def get_res_from_list(urllist, rate = 0.01, urlrate=0.05, blurbs = False, titles = False):
	"""
	takes list of urls and performs get_res_from_url on them. 
	Returns a dict of key=queryURL value = list of results
	"""
	return dict(zip(urllist,
					[get_res_from_url(url, rate=rate, blurbs = blurbs, titles = titles
						) for url in urllist if time.sleep(rate) is None]
					))  



########## SEOCOND PART / A NEW CLASS? 
"""
return info from forum threads
- username
- date of post
- thread's title [the titles may be truncated in the main pages]
"""

#TODO: check that each URL is a thread before running it.


def get_first_page(threadurl):
	"""
	Returns the first page of a thread
	from https://www.netmums.com/coffeehouse/.../.../words-words.html

	Takes a string or list of strings.
	"""
	#NOTE:n netmums puts the thread page in the URL by appended -[number] to the end of the html file.
	# if a thread's title end with  a number then -a is appended to the end to not 
	# interfere with the pagination(?). 
	# EXAMPLE URL: https://www.netmums.com/coffeehouse/family-food-recipes-555/food-tips-ideas-556/381836-how-much-per-month-feed-family-4-a-3.html
	if type(threadurl) is str:
		threadurl = re.sub('-[0-9].html', '.html', threadurl)
	elif type(threadurl) is list:
		threadurl = [re.sub('-[0-9].html', '.html', string) for string in threadurl]
	return threadurl

def resultsdict_to_urldict(results_dict):
	"""
	Return dict of dict with key from link URL, and a key in the dict
	for the queries which gave that URL.
	Also remove link URLs with different page of same thread.
	Also remove URLs which arent corresponding to a chat page (not of form https://www.netmums.com/coffeehouse/*)		

	Takes: a dictionary object which was built from the function
	get_res_from_list()
	"""

	# step 1: add the query as an item in EACH dict
	for key in results_dict.keys(): #for key in dictionary keys
		for d in results_dict[key]: #for dict in list of dict
			d['query'] = key
	
	#step 2: flatten to a list of dicts
	temp_list_of_dict = [d for each_list in results_dict.values() for d in each_list]
	
	#step 2.5 : edit the URL strings to start at page = 0. (by removing (-[0-9]).html) that group.
	for d in temp_list_of_dict:
		d['link'] = get_first_page(d['link'])
	
	
	#step 3:  build a new dict of dict with link as the key, while preserving unique query values across dicts with the same link value.
	new_dictionary = \
	{d['link']: { #for each unique link value, make a dict with key query containing a set of all query values
				'query':{d['query']} | {another_d['query'] for another_d in temp_list_of_dict if another_d['link'] == d['link']}
				} \
	for d in temp_list_of_dict}

	#step 4:  remove non-chat results.
	badkeys = []
	for key in new_dictionary.keys():
		if 'netmums.com/coffeehouse' not in key:
			badkeys.append(key)
		if '.html' not in key:
			badkeys.append(key)
			#print(key)#DEBUG
	for key in set(badkeys):
		new_dictionary.pop(key)

	return new_dictionary

### POST DATA EXTRACTION ###

def get_posts_from_page(thread_soup):
	"""
	Returns a list of soup objects corresponding to posts.
	"""
	return thread_soup.find_all('div', {'class': 
		re.compile('DesktopPostCardstyle__PostContainer-')})

def get_post_date(post_soup):
	return post_soup.find('div', {'class': 
		re.compile('DesktopPostCardHeaderstyle__PostDate-')}).text

def get_post_likes(post_soup):
	"""
	Return string containing integer of number of likes for a post.
	"""
	likes_count = post_soup.find('div', {'class': 
		re.compile('LikeCounterstyle__Container-')}).text
	return re.search('^\d+',likes_count).group()

def get_post_username(post_soup):
	"""
	Return username of poster.
	"""
	return post_soup.find('div', {'class': re.compile('__UserPseudo-')}).text

def get_post_body(post_soup):
	return post_soup.find('div', {'class': 
		re.compile('DesktopPostCardstyle__PostContent-')}).text
#UNTESTED
def extract_post(post_soup):
	"""
	takes a single post soup object
	"""
	post_dict = {'username': get_post_username(post_soup),
				'likes': get_post_likes(post_soup),
				'date': get_post_date(post_soup),
				'body': get_post_body(post_soup)}
	return post_dict

def extract_posts_from_page(thread_soup):
	"""
	Returns a list of dicts, one for each post on teh page

	Run the other post data extraction functions and return all the data
	in a dict, or something like that.

	Takes a list of posts, from get_posts_from_page()
	"""
	return [extract_post(post) for post in get_posts_from_page(thread_soup)]

### EXTRACT FROM THREAD ###
def get_thread_title(thread_soup):
	"""
	Get title of a thread soup object
	Returns a string.
	"""
	return thread_soup.find('h1', {'class': 
		re.compile('__ThreadTitle-')}).text

#works
def num_pages_in_thread(soup):
	global globalvar #DEBUG
	globalvar = soup #DEBUG
	"""
	Returns the number of pages in a thread
	Takes the soup of first page of a forum thread on netmums.com
	"""
	#get a list of page numbers shown on the page.
	#netmums has the first n pages, and then the very last page
	# (assuming there are so many pages that truncation is necessary)
	found = soup.find('div', {'class': 
		re.compile('sc-egg')})
	if found is not None:
		found = found.strings
	else:
		found = [0]

	numbers = []
	for string in found:
		try:
			numbers.append(int(string))
		except ValueError:
			pass

	return max(numbers)

#WORKS
def get_next_thread_pages(threadurl, num_pages):
	"""
	#TODO: UPDATE DESCRIPTION
	Returns a list of string URL with -[num] appended for num in 1 to num_pages.

	If applying in lambda or list, pass l = None and the function will return None.
	"""
	#TODO return error if '.html' is not found.
	if type(threadurl) is not str:
		raise TypeError('list has no values')
	if type(num_pages) is not int:
		raise TypeError('num_pages must be an integer')

	threadurl = get_first_page(threadurl)

	if num_pages == 1:
		threadurls = None
	else:
		threadurls = [re.sub('.html','-'+str(digit)+'.html',threadurl) for digit in range(2,num_pages+1)]

	return threadurls
#WORKS
def get_thread_data(threadurl, rate=0.01):
	"""
	Returns a tuple, containing thread title and a list of dicts.

	Get all the posts from an entire thread.

	Takes a thread URL
	"""
	bad_page=False
	#basic error check
	if type(threadurl) is not str:
		raise TypeError('threadurl must be a string')
	print(threadurl)#DEBUG
	#build first soup and use it to get number of pages
	first_soup = BeautifulSoup(s.get(threadurl, timeout=tout).text, 'html.parser')
	page_num = num_pages_in_thread(first_soup)
	#error checking: we jsut return blank if the URL didnt have page numbers
	if page_num == 0:
		bad_page == True

	if bad_page == False:
		#get title.
		title = get_thread_title(first_soup)
		#build remaining pages soups and concat to list
		if page_num == 1:
			all_pages = [first_soup]
		else:
			all_pages = [first_soup] + [BeautifulSoup(s.get(page, timeout=tout).text) \
					 					for page in get_next_thread_pages(threadurl, page_num) \
					 					if time.sleep(rate) is None]
	
		#iterate through the soups gathering all the info. (also flattening list)
		flat_list = [post for page in all_pages \
						for post in extract_posts_from_page(page)]
	else:
		#if the URL was bad return an empty list and a title mentioning error.
		title = 'SCRAPE_ERROR: IRRELEVANT PAGE'
		flat_list=[]

	return title, flat_list



### FINISHING TOUCHES ####
#UNTESTED
def fill_urldict(url_dict, rate = 0.063): #FIXME add sleep timer between requests.
	"""
	for each key in urldict, fill it.
	"""
	for key in url_dict:
		url_dict[key]['title'], url_dict[key]['posts'] = get_thread_data(key)
		time.sleep(rate)

	return url_dict
#UNTESTED
def get_posts_from_resultsdict(results_dict):
	"""
	Returns URLDict filled with posts.
	Takes ResultsDict
	"""
	return fill_urldict(resultsdict_to_urldict(results_dict))
