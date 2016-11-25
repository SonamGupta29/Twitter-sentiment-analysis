import re
import imp
import nltk

# Load other modules
from createDictionaries import *

# Load the dictionaries from another module
emoticonDict = getEmoticonDictionary()
wordNetDict = getWordnetDictionary()
acronymDict = getAcronymDictionary()
stopWordsDict = getStopwordDictionary()
afinnDict = getAFINNDictionary()

def removeURL(tweet):
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','', tweet)
	return tweet

def removeTargetMention(tweet):
	tweet = re.sub('@[^\s]+','',tweet)
	return tweet

def removeHashtagSign(tweet):
	tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
	return tweet

def replaceBySingleSpace(tweet):
	tweet = re.sub('[\s]+', ' ', tweet)
	return tweet

def removeNumbers(tweet):
	tweet = re.sub('[0-9]+', '', tweet)
	return tweet

def removeNonEnglishWords(tweet):
	newTweet = []
	count = 0
	for i in range(len(tweet)):
		if tweet[i] != '':
			# If any of the character matches with the normal english character then add it to result 
			chk = re.match(r'([a-zA-z0-9 \+\?\.\*\^\$\(\)\[\]\{\}\|\\/:;\'\"><,.#@!~`%&-_=])+$',tweet[i])
			if chk:
				newTweet.append(tweet[i])
				count += 1
	return newTweet, count

# This function will replace the words like
# coooooooool to coool (we will keep 3 characters as searched on internet)
def replaceRepetition(tweet):
	'''
		Can use this but not sure how to get the count of the repetative words
		re.sub(r'(.)\1+', r'\1\1', "haaaaapppppyyy")     
		'haappyy'
	'''
	specialChars = '1234567890#@%^&()_=`{}:"|[]\;\',./\n\t\r '
	count = 0
	for i in range(len(tweet)):
		x = list(tweet[i])
		if len(x) > 3:
			flag = 0
			for j in range(3, len(x)):
				if(x[j - 3] == x[j - 2] == x[j - 1] == x[j]):
					x[j - 3] = ''
					if flag == 0:
						count += 1
						flag = 1
			tweet[i] = ''.join(x).strip(specialChars)
	return tweet, count

def replaceEmoticons(tweet):
	count = 0
	for i in range(len(tweet)):
		if tweet[i] in emoticonDict:
			count += 1
			tweet[i] = emoticonDict[tweet[i]]
	return tweet, count

def expandAcronym(tweet):
	count = 0
	newTweet = []
	for i in range(len(tweet)):
		word = tweet[i].strip(specialChar)
		if word and word in acronymDict:
			count += 1
			newTweet += acronymDict[word].split(" ")
		else:
			newTweet += [tweet[i]]
	return newTweet, count

def expandNegation(tweet):
	newTweet = []
	for i in range(len(tweet)):
		word = tweet[i].strip(specialChar)
		if(word[-3:]=="n't"):
			if word[-5:]=="can't" :
				newTweet.append('can')
			else:
				newTweet.append(word[:-3])
			newTweet.append('not')
		else:
			newTweet.append(tweet[i])
	return newTweet

# Can't, not, no to negation
def replaceNegation(tweet):
	for i in range(len(tweet)):
		word = tweet[i].lower().strip(specialChar)
		if word == "no" or word == "not" or word.count("n't") > 0:
			tweet[i] = 'negation'
	return tweet

def purgeEmptySpaceTweet(tweet):
	for i in xrange(len(tweet)-1,-1,-1):
		if tweet[i] == '':
			tweet.pop(i)
	return tweet

def getPOSScore(tweet):
	count = len(tweet)
	listp = nltk.pos_tag(tweet)
	counter = 0
	nounCount, prepositionCount, adjCount = 0, 0, 0
	while counter < count:
		if(listp[counter][1] == 'NN'):
			nounCount += 1
		elif(listp[counter][1] == 'IN'):
			prepositionCount += 1
		elif(listp[counter][1] == 'JJ'):
			adjCount += 1
		counter += 1
	return nounCount, prepositionCount, adjCount

# Remove the stop words
def removeStopWords(tweet):        
	newTweet = []
	for i in range(len(tweet)):
		if tweet[i].strip(specialChar) not in stopWordsDict:
			newTweet.append(tweet[i])
	return newTweet

def processTweet(tweet):

	featureDict = dict()

	# Convert the tweet to the lower 
	tweet = tweet.lower()

	# Remove the urls from tweet
	tweet = removeURL(tweet)

	# Remove the target mention	
	tweet = removeTargetMention(tweet)

    # Remove the hashtag sign from the text
	tweet = removeHashtagSign(tweet)

	# Remove the numbers
	tweet = removeNumbers(tweet)

	# Change the multiple spaces to single spaces
	tweet = replaceBySingleSpace(tweet)

	# Remove the trailing ', ", space
	tweet = tweet.strip('\'"').strip(' ')

	# Get the words out of tweet, so split by space
	tweet = tweet.split(" ")

    # Remove the english words from the tweet and get the count of 
    # english words and count it as the feature
	tweet, count = removeNonEnglishWords(tweet)

	# Adding the feature in the dict
	featureDict['wordCount'] = str(count)

	# Replace the 	 the english words from the tweet and get the count of 
    # english words and count it as the feature
	tweet, count = replaceRepetition(tweet)

	# Adding the feature in the dict
	featureDict['repetationCount'] = str(count)

	# Replace the emoticons with the polarity
	tweet, count = replaceEmoticons(tweet)

	# Adding the feature in the dict
	featureDict['emoticonsCount'] = str(count)

	# Expand the acronyms
	tweet, count = expandAcronym(tweet)

	# Adding the feature in the dict
	featureDict['acronymCount'] = str(count)

	# Expand the negation
	tweet = expandNegation(tweet)

	# Replace the negation
	tweet = replaceNegation(tweet)

	# Remove the stop words
	tweet = removeStopWords(tweet)

	# Purge the empty tweet
	tweet = purgeEmptySpaceTweet(tweet)

	# Get the pos score
	nounCount, prepositonCount, adjectiveCount = getPOSScore(tweet)

	# Adding the feature in the dict
	featureDict['nounCount'] = str(count)
	featureDict['prepositionCount'] = str(count)
	featureDict['adjectiveCount'] = str(count)

	# Get the score from senti word net 
	wordNetScore = 0
	for word in tweet:
		if word in wordNetDict:
			wordNetScore += wordNetDict[word]

	featureDict['wordNetScore'] = str(wordNetScore)

	afinnScore = 0
	for word in tweet:
		if word in afinnDict:
			afinnScore +=  afinnDict[word]

	featureDict['afinnScore'] = str(afinnScore)
	#print featureDict	
	return tweet, featureDict