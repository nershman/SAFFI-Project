#setup.py


#PACKAGES

import pickle as pk
import gc
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import re
from joblib import Parallel, delayed
import time



import metrics_helpers as indicators
import pickle as pk
import gc
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import traceback #needed to store full error tracebacks


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk import pos_tag as part_of_speech 
from scipy.stats import pearsonr
import pandas as pd


from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize



from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pattern.en import sentiment
from pattern.en import parse, Sentence
from pattern.en import modality, mood

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm

import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn import linear_model

#SUBPACKAGES OR SOMETHING?? AKA MY OWN PACKAGES
import fuzzy_typos
import metrics_helpers as indicators




# actual setup.py thing.

setup(
    name="myPackage",
    version="0.1",
    description="A package to do stuff",
    author="Arnaud Abreu",
    author_email="arnaud.abreu.p@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "tqdm",
        "tensorflow",
    ],
    include_package_data=True,
)