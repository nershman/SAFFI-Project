from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re

#plan:
#for each vocbulary
#run process.extract for each word....


#RATIO is calcualted as:
#(lensum - ldist) / lensum where ldist is levenshtein distance


class fuzzy_typos:
	def __init__(self, vocabulary, size = 0.3):
	
		"""
		size: percent larger or smaller a word can be to be checked for a match
		"""
		# generate a dict of acceptable interval sizes.
		self.vocabulary = {i: ((1-size)*len(i), (1+size)*len(i)) for i in vocabulary}

	#TODO: UNTESTED
	def replace(self, text):
		"""
		attempt to prevent phrases from not getting matched and replace
		however it will not work with phrases that contain typos!
		"""
		for item in self.vocabulary:
			text = re.sub(item, re.sub(' ', '_', item), text)
		return text
	
	def fix_typos(self, text, replacement = None, cutoff=80):
		"""
		vocabulary : a list of strings (tokens)
		term: a string which exists in vocabulary
		replace: a string which all matche vocabularyw ill get replaced with.
		cutoff: number between 0 and 100. See fuzzywuzzy documentation for more info.
	
		Note: does not handle words at end of sentence properly (period on last word)
	
		String text will be split into sentences and processed, replacing typos in each sentence and then joined back together.
		"""

		#we will assume text has already been cleaned.
		#separate sentences
		sentence_separators = re.findall('[\n?!.]+', text) + ['']
		sentences = re.split('[\n?!.]+', text)
		#edge case thing.
		if not sentences[-1]:
			del sentences[-1]

		text = '' #reset text obj
		#process each sentence in sentences
		for ind, sent in enumerate(sentences):
			repaired = set()
			tokens = sent.split()
			for term, interval in self.vocabulary.items():
				#convert sentence to a dict, keeping only relevant words which have not been changed already
				candidates = {key:value for key, value in enumerate(tokens) if value != term and key not in repaired and interval[0] <= len(value) <= interval[1]}
				#compare using levenshtein distance (requires fuzzywuzzy and python-levenshtein)
				results = process.extractBests(term,candidates, limit=None, score_cutoff=cutoff, scorer=fuzz.ratio)		
				#update tokens
				if not replacement:
					for match in results:
						tokens[match[2]] = term
						repaired.add(match[2]) #add to the list of indexes of fixed typos
				else:
					for match in results:
						tokens[match[2]] = replacement
						repaired.add(match[2])
			if tokens:
				sent = ' '.join(tokens)
				#join sentences back together
				text += sent + sentence_separators[ind] + ' '

		return text

	def replace_words(self, text, replacement):
		"""
		This is to replace words in our texts
		word is a single string.
		"""

