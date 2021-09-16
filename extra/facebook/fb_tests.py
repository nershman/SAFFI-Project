# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-11 14:08:01
# @Last Modified by:   sma
# @Last Modified time: 2021-05-14 15:59:25


#BASIC USE OF THE PACKAGE "Search-Engines-Scraper" (https://github.com/tasos-py/Search-Engines-Scraper)
from search_engines import Google

engine = Google()
results = engine.search("my query", pages=10)
links = results.links()

#output directly:
engine.output(params)
#limit number of pages of search

#NOTE: this API will add each search to the existing set of results.

##########
## FACEBOOK##
#############


#general#
#########
import facebook_scraper
import time

#get specific posts
facebook_scraper.get_posts(post_urls = list_of_posturls, options=my_fb_options, timeout =45)

#get from page
#i think i just specify a string instead... lmfao
facebook_scraper.get_posts(account = name_of_page, options=my_fb_options, timeout =45, pages = 10)