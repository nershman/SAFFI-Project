from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re

#plan:
#for each vocbulary
#run process.extract for each word....


#RATIO is calcualted as:
#(lensum - ldist) / lensum where ldist is levenshtein distance

#from joblib import Parallel, delayed

def parallel_fix_typos():
	pass

def parallel_fix_and_replace():
	pass

def parallel_replace():
	pass

class fuzzy_typos:
	def __init__(self, vocabulary, replacement = None, size = 0.3, cleaner=None):
	
		"""
		size: percent larger or smaller a word can be to be checked for a match
		vocabulary: takes a list or set of strings. recommended that strings are tokens (no spaces)
		"""
		# generate a dict of acceptable interval sizes.
		self.vocabulary = {i: ((1-size)*len(i), (1+size)*len(i)) for i in vocabulary}

		if type(replacement) == dict:
			replacement = {to_replace:replacement_word for replacement_word, set_to_replace in replacement.items() for to_replace in set_to_replace}
		self.replacement = replacement

		self.cleaner = cleaner

	def _intersperse(self, lst, item):
		"""
		Helper function which adds underscores between a list of words.

		This is useful because string manipulation is slow in python so it's
		better to work with lists of strings and concatenate at the end.
		"""
		result = [item] * (len(lst) * 2 - 1)
		result[0::2] = lst
		return result

	def fix_typos(self, text, replacement = None, cutoff=80, cleaner = None):
		"""
		#TODO UPDATE DESCRIPTION
		vocabulary : a list of strings (tokens)
		term: a string which exists in vocabulary
		replacement: a string which all matche vocabulary will get replaced with. 
				   :OR a dict with key=replacement term, value= set of terms to replace
		cutoff: number between 0 and 100. See fuzzywuzzy documentation for more info.
	
		Note: does not handle words at end of sentence properly (period on last word)
	
		String text will be split into sentences and processed, replacing typos in each sentence and then joined back together.

		If all words are tokens (no phrases/spaces in words) then a replacement can be specified to save time over using the class, replacement.
		"""
		#if a cleaning function was specificed then use it.
		if cleaner:
			text = cleaner(text)
		elif self.cleaner:
			text = self.cleaner(text)

		#separate sentences
		sentence_separators = re.findall('[\n?!.]+', text) + ['']
		sentences = re.split('[\n?!.]+', text)
		#edge case thing.
		if not sentences[-1]:
			del sentences[-1]

		#expand dict so that keys are the thing to replace.
		#TODO: check that types are correct before doing this step.
		if not replacement:
			replacement = self.replacement #inherit from the intialization

		elif type(replacement) == dict:
			replacement = {to_replace:replacement_word for replacement_word, set_to_replace in replacement.items() for to_replace in set_to_replace}

		text = [] #reset text obj
		fixes = process.extractBests
		#process each sentence in sentences
		for ind, sent in enumerate(sentences):
			repaired = set()
			tokens = sent.split()

			for term, interval in self.vocabulary.items():
				#convert sentence to a dict, keeping only relevant words which have not been changed already
				candidates = {key:value for key, value in enumerate(tokens) if value != term and key not in repaired and interval[0] <= len(value) <= interval[1]}
				#compare using levenshtein distance (requires fuzzywuzzy and python-levenshtein)
				results = fixes(term,candidates, limit=None, score_cutoff=cutoff, scorer=fuzz.ratio)		
				#update tokens
				for match in results:
					if not replacement: 
						tokens[match[2]] = term #FIXME: can this be faster using https://wiki.python.org/moin/PythonSpeed/PerformanceTips#Initializing_Dictionary_Elements
					#if replacement is specified, replace instead of fixing typo.
					elif type(replacement) == str: 
						tokens[match[2]] = replacement
					else:
						try:
							tokens[match[2]] = replacement[term]
						except KeyError:
							tokens[match[2]] = term
					repaired.add(match[2]) #add to the list of indexes of fixed typos #FIXME https://wiki.python.org/moin/PythonSpeed/PerformanceTips#Initializing_Dictionary_Elements

			if tokens:
				#join sentences back together
				text += self._intersperse(tokens, ' ')
				text += [sentence_separators[ind], ' ']
		
		return ''.join(text)

class replacements():
	"""
	Takes a string or can also take a dict in the same format as in fix_typos for replacement.
	"""

	def __init__(self, replacements):
		if type(replacements) == list:
			self.replacements = '|'.join(replacements)
		elif type(replacements) == dict:
			self.replacements = {key: '|'.join(value) for key, value in replacements.items()}

	def replace_all(self, text):
		"""
		Only works if replacements is a dict
		"""
		try:
			for key, value in self.replacements.items():
				text = re.sub(value, key, text)
				return text
		except TypeError:
			print('TypeError: Object must be initalized with a dict of replacement:[words to replace]')

	def replace(self, replaced, text):
		"""
		Takes a phrase which will replace the terms set at initalization.
		"""
		return re.sub(self.replacements, replaced, text)

	def update_words(self, replacements):
		self.replacements = '|'.join(replacements)

	def add_word(self, word):
		"""
		take a string.
		"""
		self.replacements += '|' + word

#def replace_many(list_of_replacements, list_of_replaced, text):
	#if len(list_of_replacements)
	#this is supposed to do it for all things to replace.

