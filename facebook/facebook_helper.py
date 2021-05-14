# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-11 22:17:29
# @Last Modified by:   sma
# @Last Modified time: 2021-05-14 11:36:11


#I don;t think at the moement this has any dependencies.

#TODO fix this up to be for facebook.
def resultsdict_to_urldict(results_dict):
	"""
	TODO: update description
	Return dict of dict with key from link URL, and a key in the dict
	for the queries which gave that URL.
	Also remove link URLs with different page of same thread.
	Also remove URLs which arent corresponding to a chat page (not of form https://www.netmums.com/coffeehouse/*)		

	Takes: a dictionary object which was built from the function
	get_res_from_list()
	"""

	# step 1: add the query as an item in EACH dict
	for key in results_dict.keys(): #for key in dictionary keys
		for d in results_dict[key]: #for dict in list of dict
			d['query'] = key
	
	#step 2: flatten to a list of dicts
	temp_list_of_dict = [d for each_list in results_dict.values() for d in each_list]
	
	#step 3:  build a new dict of dict with link as the key, while preserving unique query values across dicts with the same link value.
	#by saving them to a set of values as the value of query.
	new_dictionary = \
	{d['link']: { #for each unique link value, make a dict with key query containing a set of all query values
				'query':{d['query']} | {another_d['query'] for another_d in temp_list_of_dict if another_d['link'] == d['link']}
				} \
	for d in temp_list_of_dict}

	return new_dictionary