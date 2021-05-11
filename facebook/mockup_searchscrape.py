# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-11 13:00:46
# @Last Modified by:   sma
# @Last Modified time: 2021-05-11 15:07:58

"""
First Approach:
1) search google for posts
	- this will guarantee more relevant topics, and more likely to return high-traffic posts (more comments)
2) feed post URLs into facebook scraper
	- goals: keep query source as previously w netmums scrape.
"""

from search_engines import Google
import facebook_scraper as fs
import scrapehelpers as scr

# use scraper to get links to posts on facebook

## build combo list
## this time let's put quotes around each term tho.
hazards = 



# use facebook scraper to get info

#save to csv (or pkl it if its enormopus as csv..)
#remember we want to collect the comments.
#so using pickle might be better...,.