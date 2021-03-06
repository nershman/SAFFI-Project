# SAFFI-Project
## About

This package is for reproducing the primary analysis I performed as part of my internship at IRIT in partnership with SAFFI project at INRAE. Running the scrape will scrape relevant data from both Netmums forum and Facebook and save it to a pickled dict. Running the analysis generate all graphs and tables from the report, after performin ETL on the pickled dict and cleaning. (Analysis is only performed on Netmums data, as we decided Facebook data was too noisy to find valuable information)

For more details check out [the report.](https://github.com/nershman/SAFFI-Project/blob/master/report.pdf)

# Package Setup
```
├── digitalsaffiproject
├── scraping
│	├── netmums
│	├── facebook
├── analysis
```

## Installation

Run:
```
pip install -r requirements.txt
```

## Analysis

To generate the results of the analysis,
	* open a terminal and make sure you're located inside the 'analysis' folder
	* run: python make_analysis_results.py --input [path to .pkl data file] --output [folder to save results]
	
Path to defeault pickle file: `./data/untypod_dict.pkl`

If you get an error, try 'sudo python' or changing the save directory to inside the analysis folder.
	
To genereate clean text data and select the relevant subset from it, run
```
	python clean_netmums.py --input [path to .pkl netmums data] --output [path&filename to save new .pkl] --keys [path&filename to save URLs of relevant subset]
```
	
## Scraping

### Netmums
Run:
```
	python basicscrapescript.py --blurbs-output 'path/to/picklefile.pkl' --full-output 'path/to/picklefile2.pkl'
```

### Facebook 

Four files to run

* specific_fb_groups.py - scrapes groups and saves a .pkl
* specific_fb_pages.py - scrapes pages and saves a .pkl
* extract_links_from_html.py - extracts links from google search html pages (saved locally)
* fb_searchscrape_manuallinks.py - scrapes specific facebook links (extracted from above) and saves a .pkl
