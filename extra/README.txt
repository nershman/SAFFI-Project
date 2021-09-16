############################################
### INSTRUCTIONS FOR REPLICATION #########
#########################################
NOTE: at the moment this is off the top of my head so might not be 100% accurate
-- netmums:data collection --

1) RUN basicscrapescript.py. it SAVES allposts.pkl

-- facebook: data collection --

2) RUN specific_fb_groups.py, specific_fb_pages.py, fb_searchscrape_manuallinks.py, extract_links_from_html.py
	SAVES: fb_data_searchscrape.pkl, specific_fb_groups.pkl, specific_fb_pages.pkl, manual_search_resdict.pkl
3) RUN clean_fb.py, it SAVES fb_merged_cleaned_flat.pkl

-- netmums:cleaning --

4) RUN netmums_clean_final_final. It SAVES untypod_dict.pkl and netmums_subset_keys.txt

-- netmums: analysis --

5) RUN NETMUMS-topicmining-POSTS

		
.//facebook/extract_links_from_html.py:
DESCRIPTION: takes directory of saved google search result html, exports facebook scrape data.
READS		'/Users/sma/Documents/INRAE internship/google_search_by_hand'
SAVES		open('manual_search_resdict.pkl

### IPYNB: DESCRIPTIONS & FILE dependencies######################
################################################################
# You can run this grep command to find all ipynb files that are saving or loading pkl files:
# grep -r -i -E -o open*[[:space:][:alnum:][:punct:]]+[.]pkl ./ --include "*.ipynb" --exclude '*/.ipynb_checkpoints/*'
# For other extensions similarly, just change the extension part to whatever you care about (pkl,txt, etc)
#############################################################

.//notebooks/organized.ipynb:
DESCRIPTION: descriptive statistics
READ	open('facebook/fb_data_searchscrape.pkl
READ	open('facebook/specific_fb_pages.pkl
READ	open('facebook/specific_fb_groups.pkl
READ	open('netmums/allposts_rerun.pkl
READ	open('facebook/manual_search_resdict.pkl

.//notebooks/term_counts.ipynb:
DESCRIPTION: 
READ	open('facebook/fb_data_searchscrape.pkl
READ	open('facebook/specific_fb_pages.pkl
READ	open('facebook/specific_fb_groups.pkl
READ	open('netmums/allposts_rerun.pkl
READ	open('facebook/manual_search_resdict.pkl
		
		
.//notebooks/netmums_quotes.ipynb
DESCRIPTION: 
READs	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl
READs	open('netmums_subset_keys.txt

.//notebooks/netmums_clean_final_final.ipynb:e
DESCRIPTION: builds metrics in order to define selection criterion for data, selects the 
			subset, and removes typos from text before saving to pickle
READ	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl
SAVES	open('untypod_dict.pkl
SAVES	open('netmums_subset_keys.txt

.//notebooks/NETMUMS-topicmining-THREADS.ipynb
DESCRIPTION: 
READs	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/untypod_dict.pkl
READs	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl
READs	open('netmums_subset_keys.txt

.//notebooks/netmums_clean_text_for_entire.ipynb
DESCRIPTION: 
READs	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl
saves	open('FULL_untypod_dict.pkl
		open('netmums_subset_keys.txt

.//notebooks/word2vecphrases-ENTIREDATASET.ipynb
DESCRIPTION: 
READs	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl
open('netmums_subset_keys.txt

.//notebooks/organized.ipynb
DESCRIPTION: 
READs	open('facebook/fb_data_searchscrape.pkl
READs	open('facebook/specific_fb_pages.pkl
READs	open('facebook/specific_fb_groups.pkl
READs	open('netmums/allposts_rerun.pkl
		open('facebook/manual_search_resdict.pkl

.//notebooks/term_counts.ipynb
DESCRIPTION: 
READs	open('facebook/fb_data_searchscrape.pkl
READs	open('facebook/specific_fb_pages.pkl
READs	open('facebook/specific_fb_groups.pkl
READs	open('netmums/allposts_rerun.pkl
READs	open('facebook/manual_search_resdict.pkl

.//notebooks/indicators_3_ratios.ipynb
DESCRIPTION: 
READs		open('fb_merged_cleaned_flat.pkl
READs	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl

.//notebooks/indicators_2_2016.ipynb
DESCRIPTION: 
READs		open('fb_merged_cleaned_flat.pkl
READs	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl

.//notebooks/netmums_clean_final.ipynb
DESCRIPTION: old version of clean_final_final
READs	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl

.//notebooks/temp-investigations.ipynb
DESCRIPTION: 
READs	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl

.//notebooks/NETMUMS-topicmining-POSTS.ipynb
DESCRIPTION: 
		open('/Users/sma/Documents/INRAE internship/scrape-git/facebook/untypod_dict.pkl
		open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl
		open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/netmums_subset_keys.txt

.//notebooks/word2vecphrases.ipynb
DESCRIPTION: Builds a word2vec model and look close/similar words (cosine distance) to our products and hazards
READ	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl
READ	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/netmums_subset_keys.txt

.//notebooks/FULL-NETMUMS-topicmining-THREADS.ipynb
DESCRIPTION: 
READ	open('/Users/sma/Documents/INRAE internship/scrape-git/facebook/FULL_untypod_dict.pkl
READ	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl
READ	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/netmums_subset_keys.txt
		
.//notebooks/test_typostuff.ipynb
DESCRIPTION: debug / testing of fuzzy_typo class
READ	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl

.//notebooks/indicators.ipynb
DESCRIPTION: 
READ	open('fb_merged_cleaned_flat.pkl
READ	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl

.//notebooks/clustering.ipynb
DESCRIPTION: 
READ	open('fb_merged_cleaned_flat.pkl
READ	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl

.//netmums/nbks/20210426-Investigation-1-dupe.ipynb
DESCRIPTION: 
READ	open(Path('../basicblurbs.pkl

### .py: DESCRIPTIONS & FILE DEPENDENCIES #####
# You can run this grep command to find all py files that are saving or loading pkl files:
# If you can't find the file you need, try removing "open*". 
#sma-mac:scrape-git sma$ grep -r -i -E -o open*[[:space:][:alnum:][:punct:]]+[.]pkl ./ --include "*.py" --exclude '*/.ipynb_checkpoints/*'
############################################################
		
.//facebook/fb_searchscrape_from_manuallinks.py:
DESCRIPTION: Scrape a list of facebook URLs, using a pre-made list of URLs
READs	open('manual_search_resdict.pkl
SAVES fb_data_searchscrape.pkl
		
.//facebook/specific_fb_groups.py:
DESCRIPTION: scrape specific fb groups
SAVES	open('specific_fb_groups.pkl
		
.//facebook/specific_fb_pages.py:
DESCRIPTION: scrape specific fb pages
SAVES	open('specific_fb_pages.pkl
		
.//facebook/extract_links_from_html.py:
DESCRIPTION: takes directory of saved google search result html, exports facebook scrape data.
READS		'/Users/sma/Documents/INRAE internship/google_search_by_hand'
SAVES		open('manual_search_resdict.pkl

.//facebook/clean_fb.py:
DESCRIPTION: merges the four used facebook pkl datas 
READS	open('facebook/fb_data_searchscrape.pkl
READs	open('facebook/specific_fb_pages.pkl
READs	open('facebook/specific_fb_groups.pkl
READs	open('facebook/manual_search_resdict.pkl		
SAVES	open('facebook/fb_merged_cleaned_flat.pkl -- the merged dataset for facebook

.//netmums/basicscrapescript.py:
DESCRIPTION: search netmums and extract items in two separate steps
SAVES	open('basicblurbs2.pkl -- preview text blurbs and URLS from the results page. This was used in earlier steps before
								later functions
SAVES 	open('allposts.pkl -- from URLs in basicblurbs dict, go to each URL and scrape all post data for each thread.
			
.//netmums/basicscrapescript_onlyrunsecondpart.py:
DESCRIPTION: skip first half of routine, loading basicblurbs pkl and extract threads
saves	open('allposts_rerun.pkl - same format as all_posts. TODO: remove this python file

.//facebook/fb_mockup_searchscrape.py:
DESCRIPTION

.//notebooks/make_indicator_dataframe.py:
DESCRIPTION:
READs	open('fb_merged_cleaned_flat.pkl
READs	open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl

.//notebooks/test_metrics_helpers.py:
DESCRIPTION:
		open('fb_merged_cleaned_flat.pkl
		open('/Users/sma/Documents/INRAE internship/scrape-git/netmums/allposts_rerun.pkl

.//netmums/testingfunctions.py:
DESCRIPTION:
READs	open('../basicblurbs.pkl


### PICKLE DESCRIPTIONS ###

#IMPORTANT ONES#

basicblurbs.pkl - netmums scrape data, only from the results page. queries, results, and text preview blurb.
allposts_rerun.pkl - netmums scrape data
allposts.pkl -- OLD version of allposts_rerun
fb_data_searchscrape.pkl -- facebook scrape based on links found searching google
specific_fb_groups.pkl -- fb scrape of specific facebook groups
specific_fb_pages.pkl -- fb scrape of specific facebook pages
manual_search_resdict.pkl -- extracted links from google searches, and the queries which returned that result
untypod_dict.pkl - netmums data dict cleaned, subset taken, and typos removed

#LESS IMPORTANT#

FULL_untypod_dict.pkl - netmums data, entire set, typos removed

#UNIMPORTANT#

fb_search_results.pkl - failed attempt at facebook scrape
fb_merged_cleaned_flat.pkl - the 4 relevant facebook pkls cleaned and merged into one dict
BKP-untypod_dict.pkl - backup
bkp-FULL_untypod_dict.pkl - backup
basicblurbs2.pkl - ??
basicblurbs.pkl - preliminary scrape of just blurbs
astype_copy.pkl - ???
fb_safety.pkl - ???
fb_merged_cleaned.pkl - old version of fb_merged_cleaned_flat.pkl, probably can be deleted.