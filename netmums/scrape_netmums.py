# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-04-19 15:22:28
# @Last Modified by:   sma
# @Last Modified time: 2021-04-20 12:30:13


import requests
from bs4 import BeautifulSoup

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



def netmum_basic_urls(lst):
	"""
	This function uses netmums basic URL search which is faster
	but has less functionality.
	Return list of URLS from a list of search terms.
	"""
	if not lst:
		raise TypeError('list has no values')
	return (['https://www.netmums.com/search/chat/' + terms for terms in lst])

def netmum_basic_pages():
	"""
	TODO for a url , regex the query and the create the pages to find
	"""
	pass

def getresults():
	"""
	Load URLs and the next pages
	Return a list of discussion URLS from the results
	- need to make sure that if redirected out of /chat just skip

	"""
	#there is no time limit in their robots.
requests.post(URL, data=payload, headers=hdr)
BeautifulSoup(r1.text, 'html.parser')



### TESTING
templ = netmum_urls(make_combo_list(problems, foods))


#make the request for each url with a time in between. returns a list of request objs.
#use 0.05s sleep just in case the site has some type of protection, don't want ot get blocked.
query_rq = [requests.post(url).text for url in templ[1:10] if time.sleep(0.05) is None]

# parse the html of our queries results
query_soups = BeautifulSoup(query_rq, 'html.parser')


#NOTE TO DO (MAYBE): 
# the queries only gather a limited number (100 = 10 res x 10 pages) of results
# so for now I would like to just save the number of results returned as well
# which could make it easier to figure out later.
# PHP quereis seem to be way slower (https://www.netmums.com/coffeehouse/search_result.php?config=work-692.inc)


############## CRAP
def GetDates(soup): #this works now, need to check that it works for a larger page.
    #[todo]check that arg is type bs4.BeautifulSoup
    dateList = list()
    for elm in soup.find_all('td',{"class":'basic-title'}):
        if(type(elm.find('b')) is bs4.element.Tag):
            tmpdate = elm.find('b').next_sibling
            if(len(re.findall(r'\w+\s\d+[,]\s\d\d\d\d', tmpdate)) >= 1):
                tmpdate = re.findall(r'\w+\s\d+[,]\s\d\d\d\d', tmpdate)[0]
            elif(len(re.findall(r'\w+\s\d+[,]\s\d\d\d\d', tmpdate)) >= 1):
                tmpdate = re.findall(r'\d\d[/]\d\d[/]\d\d\d\d', tmpdate)[0]
            dateList.append(tmpdate)
    return(dateList)