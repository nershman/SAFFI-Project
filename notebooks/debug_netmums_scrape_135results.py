# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-04-21 18:42:05
# @Last Modified by:   sma
# @Last Modified time: 2021-05-20 12:28:07


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

