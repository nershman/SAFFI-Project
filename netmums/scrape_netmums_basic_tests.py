# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-04-21 13:18:20
# @Last Modified by:   sma
# @Last Modified time: 2021-04-21 19:23:58

import scrapehelpers #TODO not sure if this works, maybe just not working in REPL,

### TESTING

#make list of urls (uses scrapehelpers helpoer functions)
templ = netmum_urls(make_combo_list(problems, foods))


#make the request for each url with a time in between. returns a list of request objs.
#use 0.05s sleep just in case the site has some type of protection, don't want ot get blocked.
query_rq = [requests.post(url).text for url in templ[1:3] if time.sleep(0.05) is None]


#def
#making thing to get links from search result page





#def
#soup to get the number of results




[link.find('a').get('href') for link in soup.find_all('h3', {'class', 'card__title'})]


#def
#get number of pages
#return none for 1 or fewer pages.
