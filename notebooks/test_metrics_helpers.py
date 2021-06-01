# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-06-01 05:51:45
# @Last Modified by:   sma
# @Last Modified time: 2021-06-01 14:53:09

import time
#open the facebook data
with open('fb_merged_cleaned_flat.pkl', 'rb') as f:
    fb_search = pk.load(f)

del_keys = list(fb_search.keys())
#shorten the dict to make things faster for testing.
for key in del_keys:
	if len(fb_search) > 100:
		fb_search.pop(key)

#facebook

start = time.time()
add_term_counts(fb_search, fb=True)
end = time.time()
print('termcounts execution time: ' + str(end - start))

start = time.time()
add_url_term_counts(fb_search, fb=True)
end = time.time()
print('urltermcounts execution time: ' + str(end - start))

start = time.time()
add_total_likes(fb_search, fb=True)
end = time.time()
print('totallikes execution time: ' + str(end - start))

start = time.time()
add_available_comments(fb_search, fb=True)
end = time.time()
print('availcomments execution time: ' + str(end - start))

start = time.time()
add_comment_activity(fb_search, fb=True)
end = time.time()
print('commentacitivity execution time: ' + str(end - start))

start = time.time()
add_num_unique_posters(fb_search, fb=True)
end = time.time()
print('num unique posters execution time: ' + str(end - start))

start = time.time()
add_num_urls(fb_search, fb=True)
end = time.time()
print('num urls execution time: ' + str(end - start))

start = time.time()
add_avg_comment_length(fb_search)
end = time.time()
print('avg comm length execution time: ' + str(end - start))

start = time.time()
add_avg_post_length(fb_search, fb=True)
end = time.time()
print('avg post length execution time: ' + str(end - start))

start = time.time()
add_num_comments(fb_search, fb=True)
end = time.time()
print('num comments execution time: ' + str(end - start))

start = time.time()
add_post_time(fb_search, fb=True)
end = time.time()
print('post time execution time: ' + str(end - start))

#add_post_language(fb_search, fb = True)


#netmums


with open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl', 'rb') as f:
    netmums = pk.load(f)

    
start = time.time()
add_term_counts(netmums, fb=False)
end = time.time()
print('termcounts execution time: ' + str(end - start))

start = time.time()
add_url_term_counts(netmums, fb=False)
end = time.time()
print('urltermcounts execution time: ' + str(end - start))

start = time.time()
add_total_likes(netmums, fb=False)
end = time.time()
print('totallikes execution time: ' + str(end - start))

start = time.time()
add_available_comments(netmums, fb=False)
end = time.time()
print('availcomments execution time: ' + str(end - start))

start = time.time()
add_comment_activity(netmums, fb=False)
end = time.time()
print('commentacitivity execution time: ' + str(end - start))

start = time.time()
add_num_unique_posters(netmums, fb=False)
end = time.time()
print('num unique posters execution time: ' + str(end - start))

start = time.time()
add_num_urls(netmums, fb=False)
end = time.time()
print('num urls execution time: ' + str(end - start))

start = time.time()
add_avg_comment_length(netmums)
end = time.time()
print('avg comm length execution time: ' + str(end - start))

start = time.time()
add_avg_post_length(netmums, fb=False)
end = time.time()
print('avg post length execution time: ' + str(end - start))

start = time.time()
add_num_comments(netmums, fb=False)
end = time.time()
print('num comments execution time: ' + str(end - start))

start = time.time()
add_post_time(netmums, fb=False)
end = time.time()
print('post time execution time: ' + str(end - start))

