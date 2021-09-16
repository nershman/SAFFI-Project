# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-04-21 18:42:05
# @Last Modified by:   sma
# @Last Modified time: 2021-05-20 23:08:31


import os 
os.chdir('/Users/sma/Documents/INRAE internship/scrape-git/netmums')


import scrapehelpers as scr
import scrape_netmums_basic as netmums
import pickle 
import time #not necessary but convenient
from bs4 import BeautifulSoup

import requests
s = requests.Session()

netmums.debug_requests_on()

weird_url = 'https://www.netmums.com/coffeehouse/family-food-recipes-555/food-tips-ideas-556/840648-healthy-non-processed-cheap-foodie-thread.html'

successful_keys = ['https://www.netmums.com/coffeehouse/becoming-mum-ttc-64/trying-conceive-clubs-525/1735305-january-ttc-bfp-train-17.html',
 'https://www.netmums.com/coffeehouse/being-mum-794/baby-clubs-593/1815689-babies-born-november-2018-a-23.html',
 'https://www.netmums.com/coffeehouse/pregnancy-64/preparing-baby-870/1218683-bf-vs-formula-do-both-10.html']

_, weird_nm_data = netmums.get_thread_data(weird_url)
bad_page_count = netmums.num_pages_in_thread(BeautifulSoup(s.get(weird_url).text))

soup = BeautifulSoup(s.get(weird_url).text)


### MAKING THE GETTING QUOTES STUFF.
soup = BeautifulSoup(s.get('https://www.netmums.com/coffeehouse/family-food-recipes-555/food-tips-ideas-556/840648-healthy-non-processed-cheap-foodie-thread-9.html').text)
page_soup = netmums.get_posts_from_page(soup)
post_soup_yelllow_quote = page_soup[14]
post_soup_white_quote = page_soup[13]
#the multi-quote yellow
soup = BeautifulSoup(s.get('https://www.netmums.com/coffeehouse/family-food-recipes-555/food-tips-ideas-556/840648-healthy-non-processed-cheap-foodie-thread-11.html').text)
page_soupp = netmums.get_posts_from_page(soup)
multi = page_soupp[0]
url_test = BeautifulSoup(s.get('https://www.netmums.com/coffeehouse/other-chat-514/news-12/1058671-school-names-shames-badly-parked-parents.html').text)
page_url = netmums.get_posts_from_page(url_test)
blah = page_url[0]

#NEW TEST NOW after improving scrape_netmums_basic

test_keys = [
'https://www.netmums.com/coffeehouse/family-food-recipes-555/food-tips-ideas-556/840648-healthy-non-processed-cheap-foodie-thread-9.html',
'https://www.netmums.com/coffeehouse/family-food-recipes-555/food-tips-ideas-556/840648-healthy-non-processed-cheap-foodie-thread-11.html', 
'https://www.netmums.com/coffeehouse/other-chat-514/news-12/1058671-school-names-shames-badly-parked-parents.html']
blah=[]
for url in test_keys:
	blah.append(netmums.extract_posts_from_page(BeautifulSoup(s.get(url).text)))