# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-04-19 15:01:45
# @Last Modified by:   sma
# @Last Modified time: 2021-04-30 11:31:05

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

# add the query as an item in EACH dict
for key in mydict.keys(): #for key in dictionary keys
	for dict_ in mydict[key]: #for dict in list of dict
		dict_['query'] = key

#???
#we want to convert it to a dict of key being the link value.
#but in doing this we will need to check for duplicates along the way.


#step 1: delete blurb and title and dont return error if it doesnt exist
#(dict.pop is performed in-place)
for list_of_dict in tempdict:
	for dict_ in list_of_dict:
		dict_.pop('title')
		dict_.pop('blurb')



#step 2: flatten to a list of dicts
templist = [v for k,v in fun.items()]
#[{'a': 1, 'b': 2, 'c': 3, 'd': 4}, {'a': 4, 'b': 4, 'c': 4, 'd': 4}]

#step 2.5 : edit the URL strings to start at page = 0. (by removing (-[0-9]).html) that group.
#TODO

#step 3:  build a new dict of dict with link as the key, while preserving unique query values across dicts with the same link value.
#TODO; convert this to porper vars.

#dict version (unfinished)
{resultdict['link']: {
			'query':{resultdict['query']} | {another_dict['query'] for another_dict in tempdict.values() if another_dict['link'] == resultdict['link']}
			} \
for resultdict in tempdict.values()}


#ALSO DOESNT WORK:

{resultdict['link']: {
			'query':{resultdict['query']} | {another_dict['query'] for another_dict in tempdict.values() if another_dict['link'] == resultdict['link']}
			} \
for resultlist in tempdict.values() for resultdict in resultlist}



#list version (works)
lala = \
{dict_['link']: {
			'b':{dict_['b']} | {dictt['b'] for dictt in templist if dictt['a'] == dict_['a']}
			} \
for dict_ in templist}


#ep 3: find dictss with the same value for link.
#https://stackoverflow.com/questions/20672238/find-dictionary-keys-with-duplicate-values
for dict_ in templist:
	for


#IGNORE THIS DUMB SHIT BELOW STOP BEING WEIRD AND JSUT DO IT MAN LOL LOOK AT LINK ABOVE!
#for testing:
templist = [{'a': '1', 'b': '1', 'c': '1', 'd': '1'},
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


def reorganize_resultsdict(resultsdict):
