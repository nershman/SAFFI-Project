url_dict = facehelp.resultsdict_to_urldict(results_dict)









#textdict = {key: ' '.join([str(item['text']) for item in value['data']] + \
#[' '.join([str(c['comment_text']) for c in item['comments_full']]) for item in value['data'] if item['comments_full']]
#) for key, value in fb_search.items() if print(key)}
#
#add_term_counts(fb_search, fb=False)
#
#add_url_term_counts(fb_search, fb=False)
#
#add_total_likes(fb_search, fb=False)
#
#add_available_comments(fb_search, fb=False)
#
#add_comment_activity(fb_search, fb=False)
#
#add_num_unique_posters(fb_search, fb=False)
#
#add_num_urls(fb_search, fb=False)
#
#add_avg_comment_length(fb_search)
#
#add_avg_post_length(fb_search, fb=False)
#
#add_num_comments(fb_search, fb=False)
#
#add_post_time(fb_search, fb=False)
#

#opendata


academic_sample = 'In sum, all textual analyses are fraught with difficulty and disagreement, and LD is no exception. There is no agreement in the field as to the form of processing (sequential or nonsequential) or the composition of lexical terms (e.g., words, lemmas, bigrams, etc.); and even a common position with regard to the distinction between the terms lexical diversity, vocabulary diversity, and lexical richness remains unclear (Malvern et al., 2004). In this study, we do not attempt to remedy these issues. Instead, we argue that the field is sufficiently young to be still in need of exploring its potential to inform substantially. Thus, we include in our analyses the most sophisticated indices of LD that are currently available.'

print('Sample MTLD:', ld.mtld(academic_sample.split()))
print('Sample HD-D:', ld.hdd(academic_sample.split()))

