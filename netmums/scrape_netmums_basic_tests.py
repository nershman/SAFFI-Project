# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-04-21 13:18:20
# @Last Modified by:   sma
# @Last Modified time: 2021-04-21 16:36:56

import scrapehelpers #TODO not sure if this works, maybe just not working in REPL,

### TESTING

#make list of urls (uses scrapehelpers helpoer functions)
templ = netmum_urls(make_combo_list(problems, foods))


#make the request for each url with a time in between. returns a list of request objs.
#use 0.05s sleep just in case the site has some type of protection, don't want ot get blocked.
query_rq = [requests.post(url).text for url in templ[1:3] if time.sleep(0.05) is None]


#def
#making thing to get links from search result page

soup = BeautifulSoup(query_rq[0], 'html.parser')



#def
#soup to get the number of results

res = int(re.search('\d+\s', soup.find('p', {'class':'search-results__count'}).text))