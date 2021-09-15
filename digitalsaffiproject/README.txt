Package Setup

├── digitalsaffiproject
├── scraping
│	├── netmums
│	├── facebook
├── analysis


##############
#Installation#
##############
Run:
	pip install -r requirements.txt


##########
#Analysis#
##########

To generate the results of the analysis,
	- open a terminal and make sure you're located inside the 'analysis' folder
	- run: python make_analysis_results.py --input [path to .pkl data file] --output [folder to save results]
	
Path to defeault pickle file: ./data/untypod_dict.pkl

If you get an error, try 'sudo python' or changing the save directory to inside the analysis folder.
	
To genereate clean text data and select the relevant subset from it, run
	python clean_netmums.py --input [path to .pkl netmums data] --output [path&filename to save new .pkl] --keys [path&filename to save URLs of relevant subset]