# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-04-19 15:01:45
# @Last Modified by:   sma
# @Last Modified time: 2021-05-07 08:38:12

#test combo list
from itertools import repeat, permutations
l1 = ["a", "b"]
l2 = ["c", "d"]

temp = list(product(l1,l2,l2))
make_combo_list(l1,l2)
make_combo_list(l1,["c","d",2])
make_combo_list(l1,2)
make_combo_list(l1,"oxo")


#second beautifull soup..

import re

html = requests.get('https://www.netmums.com/coffeehouse/becoming-mum-ttc-64/early-pregnancy-signs-symptoms-537/797571-8-weeks-pregnant-got-shingles.html').text
soup = BeautifulSoup(html, 'html.parser')
temp_posts = 

temp_post = temp_posts[0]


# convert the resultsdict to a way which we can remove duplicates.import os
import os
os.chdir('/Users/sma/Documents/INRAE internship/scrape-git/netmums/nbks')
mydict = pk.load(open('../basicblurbs.pkl','rb'))

tempdict = mydict['https://www.netmums.com/search/chat/Chemical contaminants formula']
tempdict = {'https://www.netmums.com/search/chat/Chemical contaminants formula':tempdict}

[key for key in mydict.keys()]

##############
#add query as an attribute to each result-dict.
###############
###

#testing the final function...
#subset of dictionary
tempdict = {key:mydict[key] for key in list(mydict.keys())[0:8]}

temp = get_posts_from_resultsdict(tempdict)

globalvar.find('fieldset', {'class': 
		re.compile('^ActiveThreadsstyle__')}).find_next_sibling().strings

#IGNORE THIS DUMB SHIT BELOW STOP BEING WEIRD AND JSUT DO IT MAN LOL LOOK AT LINK ABOVE!
#for testing:
tempplist = [{'a': '1', 'b': '1', 'c': '1', 'd': '1'},
{'a': '4', 'b': '4', 'c': '4', 'd': '4'},
{'a': '1', 'b': '1', 'c': '1', 'd': '1'},
{'a': '3', 'b': '0', 'c': '2', 'd': '4'},
{'a': '1', 'b': '2', 'c': '1', 'd': '1'},
{'a': '4', 'b': '99', 'c': '4', 'd': '4'}]

#a -> 'link'
{dict_['a']:{k:v+vv for k,v in dict_.items() for _,vv in dict_.items()} for dict_ in templist}



for dict_ in templist:



#merge them together somehow.. not sure how. fuck.
#maybe here: https://stackoverflow.com/questions/20672238/find-dictionary-keys-with-duplicate-values

#lol... make new dict from list dict with evberything donme

#return as a dict of dicts based on a specific value
{dict_['a']: {'b':{dict_['b']} | {dictt['b'] for dictt in templist if dictt['a'] == dict_['a']}} \
for dict_ in templist}



# find values that are the same
for k,v in fun.items():
	for kk,vv in v.items():
		()

#trying to figure out how to merge...
fun


for k,v in fun.items():
	k:{kk:vv for kk,vv in v.items()}

#YAY IT WORKS
{k:{kk:vv for kk,vv in v.items()} for k,v in fun.items()}

{k:{vv:kk for kk,vv in v.items() for kkk,vvv in v.items() if kk==kkk and vv==vvv} for k,v in fun.items()}

#this works for repalcing w sets.
{k:{kk:{vv,vvv} for kk,vv in v.items() for kkk,vvv in v.items()} for k,v in fun.items()}


#dict comprehension to construct new dict
{k:v for k,v in fun.items()}


####

tempd = {}
for obj in get_posts_from_page(thread_soup):
	#add each thing 


####
first_soup.find('div', {'class': re.compile('__ThreadTitle-')}).text
