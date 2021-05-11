# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-11 14:08:01
# @Last Modified by:   sma
# @Last Modified time: 2021-05-11 14:10:25


#BASIC USE OF THE PACKAGE "Search-Engines-Scraper" (https://github.com/tasos-py/Search-Engines-Scraper)
from search_engines import Google

engine = Google()
results = engine.search("my query")
links = results.links()
#TODO: i think it keeps going until it gets all the links.. 0.0
#I only want to get like top ten pages or something like that.
