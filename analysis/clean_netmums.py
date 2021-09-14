#!/usr/bin/env python
# coding: utf-8

# In this notebook, we:
# * select a subset based on indicators derived from the text and related data
#     * we do not process the text as thorougly at this step because we are only working with entire threads at this point, so it should be expected that the vocabulary we are searching for appear at least once in each thread.
# * export a set of the keys (URLs) which are selected for the subset
# * process text of relevant subset so that it can be better evaluated in further steps.



### RUN FILE WITH:
#
# python clean_netmums.py --blurbs-output 'path/to/picklefile.pkl' --full-output 'path/to/picklefile2.pkl'
#




import metrics_helpers as indicators
import pickle as pk
import gc
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='allposts.pkl')
parser.add_argument('-k', '--keys', default='netmums_subset_keys.txt')
parser.add_argument('-o', '--ouput', default='untypod_dict.pkl')
args = parser.parse_args()

input_file = args.input # '/Users/sma/Documents/INRAE internship/scrape-git/facebook/untypod_dict.pkl'
key_file = args.keys  #'/Users/sma/Documents/INRAE internship/temppics/'
output_file = args.output  #'/Users/sma/Documents/INRAE internship/temppics/'

# # Create Indicators

# In[3]:


def dt_to_int(dt): #datetime to integer
    return dt.astype('int')/(10**9)


# In[4]:


#netmums

with open(input_file, 'rb') as f:
    netmums = pk.load(f)

    
nm_ind = indicators.indicators(netmums, fb=False)
#this one takes long, around 20 seconds I think.

nm_ind.add_term_counts()
nm_ind.add_url_term_counts()
nm_ind.add_total_likes()
nm_ind.add_available_comments()
nm_ind.add_comment_activity()
nm_ind.add_num_unique_posters()
nm_ind.add_num_urls()
nm_ind.add_avg_post_length()
nm_ind.add_post_time()
nm_ind.add_lexical_richness()
nm_ind.add_term_distance_simple()

netmums = nm_ind.results_dict


# In[5]:


distance_data = [item.get('term_distance_simple') for item in netmums.values()]
pd.Series([item is not None for item in distance_data]).value_counts()


# In[6]:


#now plot distributino of values.
plt.hist([item for item in distance_data if item is not None], bins=500)
plt.title('min distance of product and hazard terms by document')
plt.xscale('log')


# In[7]:


pd.Series(distance_data).sort_values().value_counts(sort=False)


# In[8]:


#now plot distributino of values.
plt.hist([item for item in distance_data if item is not None], bins=200)
plt.title('min distance of product and hazard terms by document')
#plt.xscale('log')


# In[9]:


nmdf = pd.DataFrame.from_dict(netmums).transpose()
nmdf = nmdf.drop(columns=['posts', 'term_counts', 'url_term_counts', 'query'])


# # Pre-processing
# 

# ## move term counts to separate columns

# ### netmums

# In[11]:


terms = list([n for n in netmums.values()][0]['term_counts'].keys())

for term_key in terms:
    d = {url_key: value['term_counts'][term_key] for url_key, value in netmums.items()}
    nmdf['term_counts_' + term_key] = nmdf.index.map(d)
    
for term_key in terms:
    d = {url_key: value['url_term_counts'][term_key] for url_key, value in netmums.items()}
    nmdf['url_term_counts_'+ term_key] = nmdf.index.map(d)


# In[12]:


nmdf = nmdf.reset_index()
nmdf = nmdf.rename(columns={"index":"url"})
nmdf


# In[13]:


gc.collect()


# ## Create two term totals
# combine counts for terms which are hazards and terms which are products.

# In[14]:


#choose terms to count in totals
term_count_df = nmdf[nmdf.columns[pd.Series(nmdf.columns).str.startswith('term_counts_')]]

products = term_count_df.columns[[5,12,13,17,33, 34, 35, 44,45,57, 58, 65,66,68, 69,70,78, 79,80, 85]]
hazards = term_count_df.columns[[1,2,3,9,10,11,14,15,16,18,
                                 19,20,21,22,23,24,25,26,27,
                                 28,30,31,37,38,39,40,
                                 41,42,43,46,47,48,49,50,51,
                                 52,53,54,55,56,59,60,61,62,
                                 63,64,67,71,72,73,74,75,76,
                                 77,81,82,83,84]]


# In[15]:


# create the totals of terms in two categories.

# In[17]:


#netmums
#add totals
term_count_df = nmdf[nmdf.columns[pd.Series(nmdf.columns).str.startswith('term_counts_')]]

products = term_count_df.columns[[5,12,13,17,33, 34, 35, 44,45,57, 58, 65,66,68, 69,70,78, 79,80, 85]]
hazards = term_count_df.columns[[1,2,3,4,6,7,8,9,10,11,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,36,37,38,39,40,41,42,43,46,47,48,49,50,51,52,53,54,55,56,59,60,61,62,63,64,67,71,72,73,74,75,76,77,81,82,83,84]]


term_count_hazard_df = term_count_df[hazards]
term_count_product_df = term_count_df[products]

url_term_count_df = nmdf[nmdf.columns[pd.Series(nmdf.columns).str.startswith('url_term_counts_')]]

url_term_count_hazard_df = url_term_count_df['url_' + hazards]
url_term_count_product_df = url_term_count_df['url_' + products] 


nmdf['total_term_count'] = term_count_df.sum(axis=1) #make sure there's no s because we can easily filter the non totals then with contains.
nmdf['total_url_term_count'] = term_count_df.sum(axis=1)

nmdf['total_term_count_hazard'] = term_count_hazard_df.sum(axis=1) #make sure there's no s because we can easily filter the non totals then with contains.
nmdf['total_url_term_count_hazard'] = url_term_count_hazard_df.sum(axis=1)

nmdf['total_term_count_product'] = term_count_product_df.sum(axis=1) #make sure there's no s because we can easily filter the non totals then with contains.
nmdf['total_url_term_count_product'] = url_term_count_product_df.sum(axis=1)


# ## Examine the Pandas Dataframes.

# In[18]:


#we want to plot these
nmdf[nmdf.columns[pd.Series(nmdf.columns).str.contains('term_counts_') == 0]].drop(['url','title'],axis=1)


# In[19]:


gc.collect()


# ## Post-2016 Subset

# In[20]:


from datetime import datetime


# In[21]:


time_cutoff = datetime(2016,1,1, 0, 0, 0, 0)


# In[22]:


nmdf = nmdf.loc[nmdf.post_time >= time_cutoff]


# # Make Indicators Relative so More Easily Interpreted.

# In[23]:


#TODO adjust this for the new indicators...
#will involve some playing around / investigating I guess.

nmdf = nmdf.copy()
#so we don't get an error for operating on a slice or sth


# In[24]:


nmdf['total_term_count'] = nmdf['total_term_count'] / (nmdf['avg_post_length'] + 1)
nmdf['total_term_count_hazard'] = nmdf['total_term_count_hazard'] / (nmdf['avg_post_length'] + 1)
nmdf['total_term_count_product'] = nmdf['total_term_count_product'] / (nmdf['avg_post_length'] + 1)
#FIXME divide by available comments insteand of average post length (for netmums only.)

nmdf['num_unique_posters'] = nmdf['num_unique_posters'] / (nmdf['available_comments'] + 1)


# ## Modify Time so We can Graph It.

# In[25]:

nmdf['int_post_time'] = dt_to_int(nmdf.post_time.dropna()) 


# # Pair Plots
# 
# Take a sample of 500 points and plot the pair plots of our indicators.
# Datetime could not be plotted, so it's converted to int. We can estimate on the graph what the corresponding date is by looking at quantiles.
# Lexical richness uses a measure called MTLD, which should be less biased for short documents than the most common TTR (Type-Token Ratio)
# 

# In[26]:


nm_indicators = ['total_term_count',
'total_term_count_hazard',
'total_term_count_product',
'num_unique_posters',
'avg_post_length',
'lexical_richness', 
'term_distance_simple']


# ## Netmums

# In[27]:


nmdf[nmdf.columns[nmdf.columns.isin(nm_indicators)]].loc[nmdf.term_distance_simple.isnull()]


# In[28]:


nmdf[nmdf.columns[nmdf.columns.isin(nm_indicators)]].loc[nmdf.term_distance_simple.notnull()]


# ### Graph of Only Term Counts

# In[29]:


view = nmdf.loc[nmdf.total_term_count_hazard <1].loc[nmdf.total_term_count_product <1]
plt.scatter(view.total_term_count_hazard, view.total_term_count_product)


# In[30]:


a = nmdf.dropna()['term_distance_simple']
b = nmdf.dropna()['lexical_richness']


# In[31]:


tempdf = pd.DataFrame([list(a),list(b)]).transpose()
tempdf = tempdf.rename(columns={0:'term', 1:'lex'})


# In[32]:


blah = nmdf.term_distance_simple / ( nmdf.avg_post_length * nmdf.available_comments)


# In[33]:


a = blah.dropna()

tempdf = pd.DataFrame([list(a),list(b)]).transpose()
tempdf = tempdf.rename(columns={0:'term', 1:'lex'})


# # Final Set of Criterion
# 
#  * year >= 2016
#  * non-zero occurences of both a product and a hazard term
#  * term distance between product and hazard is below the 95th percentile.
#  
#  * if this approach proves not to yield results, we can instead attempt to take a subset based on the subforum which results occur in !

# # Basic Descriptive Statistics
# 

# ## Netmums

# In[34]:


#get the sub foruma nd sub sub forum that results are in 
site_area = view.url.str.extract('netmums.com\/coffeehouse\/(?P<subforum>[^\/]+)\/(?P<subsubforum>[^\/]+)')


# In[35]:


site_area['subforum'].value_counts().plot.barh()


# In[36]:


site_area['subsubforum'].value_counts().plot.barh(figsize=(5,15))


# In[37]:


plt.hist([i for i in view.term_distance_simple if i], bins=200)
plt.yscale('log')


# In[38]:


np.quantile([i for i in view.term_distance_simple if i], [0.25,0.5,0.75, 0.95])


# ### Export the Keys for our Desired Subset.

# In[39]:


#export the keys #TODO
relevant_keys = nmdf.loc[nmdf.total_term_count_hazard > 0].loc[nmdf.total_term_count_product > 0].loc[nmdf.post_time >= time_cutoff].loc[nmdf.term_distance_simple <= 491]


# In[40]:


with open(key_file,'w') as f:
	f.writelines([i + "\n" for i in list(relevant_keys.url)])


# # Clean Text
# We now construct our subset form the desired keys and then process the text.
# * TODO: clean the text before we run it through the next steps. By removing hyphens, upper cases, etc.
#     * but not lemmatization, unless we also lemmatize our lists of words to search for!!!
# 
# * remove typos of relevant words using Levenshtein Distances
# * replace tokens for specific foods and brands with their category, after compiling lists of these terms using word2vec
#     * replace tokens for all types of fruits with fruit
#     * replace tokens for all types of vegetable with vegetable
#     * replace tokens for all types of grains with "cereal" (???) should I??

# ### Setup: Define Cleaning Function

# In[41]:


#note if we replace hyphens with spaces at this step we may have
#issues fully removing URLs. Let's remove hyphens later in the pipeline.

#lowercasing is implemented as an option within the package.
import re #TODO: is there a better way of doing this? my pakage already imports re.
def clean(text):
    #lowercase
    text = text.lower()
    #remove URLs.
    reg = '\S+.(?:co|net|tv|org|edu|gov)\S*'
    text = re.sub(reg, '', text)
    
    return text


# ### Setup: Create Lists for Relevant Terms

# In[42]:


baby_formula = ['nutramigen',
 'neocate',
 'powdered milk',
 'infasoy',
 'comfort milk', #brand name which people dont write formula alongside
 'sma' 
]

baby_cereal = ['baby rice '#this one is really useful / important. idk how exactly to handle it.
'rusks' #a cereal food for babies to teethe with
]

cereal = ['cornflakes',
'muesli',
'bran flakes',
'cheerios',
'shreddies',
'weetabix',
'ready brek',
'rice pudding',
'rice'
]

fruit = ['banana',
'berries',
'blueberries',
'raisins',
'apples',
'pear',
'strawberries',
'pineapple', 
'raspberries',
'mango', 
'prunes', 
'grapefruit']

veg = ['mushroom', 
'red_pepper',
'green_beans', 
'courgette', 
'broccoli', 
'tomato',
'parsnips', 
'greens', 
'potato', 
'carrots',
'broccoli',
'cucumber', 
'peas', 
'tomatoes', 
'sweet_potato',
'sweetcorn', 
'corn', 
'spinach', 
'cauliflower',
'butternut squash', 
'beetroot',
'squash']


# In[43]:


foodwords = [
#infant formula
"formula","baby formula", "bottle-fed", "bottle",
#sterizlized vegetable mixed with fish
"veggie baby food","vegetable baby food",
"veg puree", "veg purée",
#fresh fruit puree mildly processed
"fruit puree","fruit baby food", "fruit purée", "applesauce",
#infant cereals
"cereal for baby", "cereal", "porridge", "oats", "oatmeal",
#other
"jar food", "baby food", "jarred", "premade food", "puree", "purée", "jarred food"
,"yoghurt", "pudding"]

hazardwords = ["Chemical contaminants",#ENDOCRINE DISRUPTOR
"Endocrine disruptor","endocrine","estrogen",#end
#FOOD PRESERVATIVES, SWEETENERS AND ADDITIVES
"preservatives","sweeteners","additives", #end
"Pesticides",#VETERINARY DRUGS
"Veterinary drugs","animal drugs","vet drugs", #end
#GMO
"GMO", "genetically modified",#end
"Metals","Mycotoxin",#BISPHENOL A
"Bisphenol","BPA", #end
#FURAN - removed because nothing related to this returns results
#DON (note that this acronym nobody uses and all results are from words like "don't")
"deoxynivalenol","vomitoxin",#end
#DIOXIN AND PCB
"Dioxin","PCB","biphenyls",#end
#MOH
"MOH","hydrocarbons","saturated hydrocarbons","MOAH","aromatic hydrocarbons",#end
"Nitrates",
#ACRYLAMID
"Acrylamide",
"phthalates",
#MICROBIOLOGIC CONTAMINANTS
"Microbiologic contaminants","spores","mold","mould","virus","microbes","contaminated",#end
"Salmonella","Campylobacter","Listeria",
#ECOLI
"EColi",
"Cronobacter",
"Histamine",
#other bacteria
"bacteria",#end
"Virus",
"Parasites",
#UNRELATED BUT MAYBE USEFUL?
"carcinogen","chemicals", "toxic", "toxin", "poisonous", "fungus", "food poisoning", "hazard","EFSA","European Food Safety Authority"]


# ### Setup: Define Functions

# In[44]:


import fuzzy_typos


# In[45]:


typos_to_fix_or_replace = veg + fruit + cereal + baby_formula + baby_cereal
typos_to_fix_or_replace = {word for phrase in typos_to_fix_or_replace for word in phrase.split()} #typos to fix and single tokens to replace

replacements_dictionary = {'vegetable':veg, 'fruit':fruit, 'cereal':cereal, 'baby cereal':baby_cereal, 'baby formula': baby_formula}

remaining_words_to_replace = {key:[item for item in value if ' ' in item] for key, value in replacements_dictionary.items()}


# In[46]:


fix_and_replace_tokens = fuzzy_typos.fuzzy_typos(typos_to_fix_or_replace, replacements_dictionary, cleaner = clean)
replace_phrases = fuzzy_typos.replacements(remaining_words_to_replace)


# ### Setup: Parallel Processing

# In[47]:


from joblib import Parallel, delayed
import time


# In[48]:


keys = list(relevant_keys.url)
num_keys = len(keys)
num_lists = 20 #how many instances will be split for parallel processing
list_of_list_of_keys = [keys[slice(i,num_keys,num_lists)] for i in range(num_lists)]

def get_small_dict(list_of_keys): #we give process small dicts because o.w. the whole dict (a global) will get duplicated in each instance
    return {key: netmums[key] for key in list_of_keys}

def process(typofixer,replacer,small_dict): #now process takes two objects
    #approx 1.5x slower than the text_dict way.
    #THE RELEVANT THINGS:
    #netmums[blah]['title']
    #netmums[blah]['posts'][n]['body']
    #netmums[blah]['posts'][n]['quotes_w']
    #netmums[blah]['posts'][n]['quotes_y']['text']
    for key, value in small_dict.items():
        small_dict[key]['title'] = typofixer.fix_typos(small_dict[key]['title'])
        for ind, item in enumerate(small_dict[key]['posts']): #a list of dicts
            if item['body']:
                small_dict[key]['posts'][ind]['body'] = replacer.replace_all(typofixer.fix_typos(item['body']))
            if item['quotes_w']:
                for qind, quote in enumerate(item['quotes_w']):
                    small_dict[key]['posts'][ind]['quotes_w'][qind] = replacer.replace_all(typofixer.fix_typos(quote))
            if item['quotes_y']:
                for qind, quote in enumerate(item['quotes_y']):
                    small_dict[key]['posts'][ind]['quotes_y'][qind]['text'] = replacer.replace_all(typofixer.fix_typos(quote['text']))     
    return small_dict


# In[49]:


#generate dicts which we will feed into the parallel processing
#if we feed the entire dict in and generate them from within it, 
#the whole dict will get duplicated  many times wasting memory.

list_of_small_dict = [get_small_dict(i) for i in list_of_list_of_keys]


# ### Finally Running It

# In[50]:


start = time.time()
results = Parallel(n_jobs=-1)(delayed(process)(fix_and_replace_tokens,replace_phrases,i) for i in list_of_small_dict)
end = time.time()
print('default food words time: ' + str(end - start),)


# In[51]:


untypod_dict = {key:value for dictionary in results for key,value in dictionary.items()}


# In[52]:


with open(output_file, 'wb') as f:
    pk.dump(untypod_dict, f)


# In[ ]:




