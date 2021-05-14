# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-11 14:08:01
# @Last Modified by:   sma
# @Last Modified time: 2021-05-14 13:34:31


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

from facebook_scraper import *
import time

start = time.time()
posts = list(get_posts(
    post_urls=["https://m.facebook.com/story.php?story_fbid=1517650235239786&id=144655535872603"],
    cookies="cookies.txt",
    options={"comments":500}
))
print(f"{len(posts[0].get('comments_full'))} comments extracted in {round(time.time() - start)}s")

### get comments
start = time.time()
posts = list(facebook_scraper.get_posts(
    post_urls=["https://m.facebook.com/story.php?story_fbid=1517650235239786&id=144655535872603"],
    cookies="cookies.txt",
    options={"comments":True}
))
print(f"{len(posts[0].get('comments_full'))} comments extracted in {round(time.time() - start)}s")




# get specific post#
####################
from facebook_scraper import get_posts
import pprint
posts = list(get_posts(post_urls=["https://m.facebook.com/story.php?story_fbid=1931709990319458&id=285708024919671"]))
pprint.pprint(posts)


# get posts from a page #
#########################