# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-04-21 18:42:05
# @Last Modified by:   sma
# @Last Modified time: 2021-05-07 11:38:23

import scrapehelpers as scr
import scrape_netmums_basic as netmums
import pickle 
import time #not necessary but convenient

foods = scr.get_foods()
concerns = scr.get_concerns()


mylist = scr.make_combo_list(concerns, foods)

myurls = netmums.build_search_urls(mylist)

netmums.debug_requests_on()

start = time.time()


myresultsdict = netmums.get_res_from_list(myurls, titles=False, blurbs = False)

end = time.time()

print('execution time: ' + str(end - start), '\nsaving dict to pickle...')
#save it..

filehandler = open('basicblurbs.pkl', 'wb')  
pickle.dump(myresultsdict, filehandler)


#change structure so that its a list of dict containing URL, etc. and then theres an added key 'queries' which is a set of query eis was returned in .
start = time.time()
final_data = netmums.get_posts_from_resultsdict(myresultsdict)
end = time.time()
print('execution time: ' + str(end - start), '\nsaving dict to pickle...')

filehandler = open('allposts.pkl', 'wb')  
pickle.dump(final_data, filehandler)
#my_huge_data_dict = get_posts_from_list(list(myresultsdict.key('url')))
#data dict will contain post content, username, title...