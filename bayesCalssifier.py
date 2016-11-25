import re
import sys
import random
from itertools import izip
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB as NB
from scipy.sparse import *
import numpy as np
from sklearn.utils import shuffle

# Load our modules
from processTweets import *
from ark import *
from createDictionaries import *

emoticonDict = getEmoticonDictionary()
acronymDict = getAcronymDictionary()
stopWordsDict = getStopwordDictionary()
wordnetDictionary = getWordnetDictionary()
# Dictionary of words
wordDictionary = dict()
# positive Tweet Count
positiveCount = 0
# Negative Tweet Count
negativeCount = 0
# Neutral Tweet Count
neutralCount = 0
# Total Tweet Count
totalTweetCount = 0
# All words list
wordList = []
# Word count
wordCount = 0
# Probabilities
positiveProbability = 0
negativeProbability = 0
neutralProbability = 0

def createDictionary(fileName):
	global wordDictionary, wordList, positiveCount, negativeCount, neutralCount, totalTweetCount, wordCount
	with open(fileName, 'r') as f:
		for line in f:			
			totalTweetCount += 1
			line = line.rstrip().rstrip('\n')
			fields = re.split(r'\t+', line)
			polarity = fields[0].strip()
			tweetText = fields[1].strip()
			if "positive" == polarity:
				positiveCount += 1
			elif "negative" == polarity:
				negativeCount += 1
			else:
				neutralCount += 1

			tokens = re.split(' ', tweetText)
			for token in tokens:
				wordDictionary[token] = 0 

	#Read Features(NON_ENG,REPEAT,EMOTICON,ACRONYM and WN_SCORE) here and them to dictionary)
	wordDictionary['NON_ENG'] = 0
	wordDictionary['REPEAT'] = 0
	wordDictionary['EMOTICON'] = 0
	wordDictionary['ACRONYM'] = 0
	wordDictionary['WN_SCORE'] = 0
	wordDictionary['NN'] = 0
	wordDictionary['IN'] = 0
	wordDictionary['JJ'] = 0

	#sort dictionary
	wordList = sorted(wordDictionary)	

	#Assign index number to each word in dictionary
	for word in wordList:
	    wordDictionary[word] = wordCount
	    wordCount += 1
	

# Initial processing
def init():

	# Create the dictionaries
	createDictionary('processedTweets.txt')
	train_X, train_Y, positiveMatrix, negativeMatrix, neutralMatrix = generateFeatures()

	# Get the probabilities of the all three probabilities
	global positiveCount, negativeCount, neutralCount, totalTweetCount
	positiveProbability = float(positiveCount) / float(totalTweetCount)
	negativeProbability = float(negativeCount) / float(totalTweetCount)
	neutralProbability = float(neutralCount) / float(totalTweetCount)

	# Take the input from the user
	tweet = raw_input("Enter the tweet : ")

	classifyTweet(tweet, positiveMatrix, negativeMatrix, neutralMatrix)


def generateFeatures():

	global positiveCount, negativeCount, neutralCount, wordCount, wordDictionary, wordList

	# dok_matrix -> Dictionary Of Keys based sparse matrix. This is an efficient structure for 
	# constructing sparse matrices incrementally.
	positiveMatrix = dok_matrix((positiveCount, wordCount))
	negativeMatrix = dok_matrix((negativeCount, wordCount))
	neutralMatrix = dok_matrix((neutralCount, wordCount))

	pos = 0
	neg = 0
	neu = 0	

	with open("processedTweets.txt", "r") as tweetFile, open("featureVectors.txt", "r") as featureFile: 
		for tweetText, featureVector in izip(tweetFile, featureFile):    		
			tweetText = tweetText.strip().strip('\n')
			featureVector = featureVector.strip().strip('\n')
			# Split the feature vector
			featureList = re.split(' ', featureVector)
			# get the polarity and tweettext
			polarity, tweetText = re.split(r'\t+',tweetText)

			tokens = re.split(' ', tweetText)

			if "positive" == polarity:
				for token in tokens:
					positiveMatrix[pos, wordDictionary[token]] = 1
				positiveMatrix[pos, wordDictionary['NON_ENG']] = featureList[0]
				positiveMatrix[pos, wordDictionary['REPEAT']] = featureList[1]
				positiveMatrix[pos, wordDictionary['EMOTICON']] = featureList[2]
				positiveMatrix[pos, wordDictionary['ACRONYM']] = featureList[3]
				positiveMatrix[pos, wordDictionary['WN_SCORE']] = featureList[4]
				positiveMatrix[pos, wordDictionary['NN']] = featureList[5]
				positiveMatrix[pos, wordDictionary['IN']] = featureList[6]
				positiveMatrix[pos, wordDictionary['JJ']] = featureList[7]
				pos += 1

			elif "negative" == polarity:
				for token in tokens:
					negativeMatrix[neg, wordDictionary[token]] = 1
				negativeMatrix[neg,wordDictionary['NON_ENG']] = featureList[0]
				negativeMatrix[neg,wordDictionary['REPEAT']] = featureList[1]
				negativeMatrix[neg,wordDictionary['EMOTICON']] = featureList[2]
				negativeMatrix[neg,wordDictionary['ACRONYM']] = int(featureList[3])
				negativeMatrix[neg,wordDictionary['WN_SCORE']] = featureList[4]
				negativeMatrix[neg,wordDictionary['NN']] = featureList[5]
				negativeMatrix[neg,wordDictionary['IN']] = featureList[6]
				negativeMatrix[neg,wordDictionary['JJ']] = featureList[7]
				neg += 1

			else:
				for token in tokens:
					neutralMatrix[neg, wordDictionary[token]] = 1
				neutralMatrix[neu, wordDictionary['NON_ENG']] = featureList[0]
				neutralMatrix[neu, wordDictionary['REPEAT']] = featureList[1]
				neutralMatrix[neu, wordDictionary['EMOTICON']] = featureList[2]
				neutralMatrix[neu, wordDictionary['ACRONYM']] = featureList[3]
				neutralMatrix[neu, wordDictionary['WN_SCORE']] = featureList[4]
				neutralMatrix[neu, wordDictionary['NN']] = featureList[5]
				neutralMatrix[neu, wordDictionary['IN']] = featureList[6]
				neutralMatrix[neu, wordDictionary['JJ']] = featureList[7]
				neu += 1

	pMatrix = positiveMatrix
	nMatrix = negativeMatrix
	neMatrix = neutralMatrix

	# Compress the rows
	positiveMatrix.tocsr()	
	negativeMatrix.tocsr()
	neutralMatrix.tocsr()

	positiveMatrix = hstack([positiveMatrix, csr_matrix([[0],] * positiveCount)])
	negativeMatrix = hstack([negativeMatrix, csr_matrix([[1],] * negativeCount)])
	neutralMatrix = hstack([neutralMatrix, csr_matrix([[2],] * neutralCount)])

	finalMatrix = vstack([positiveMatrix,negativeMatrix])
	finalMatrix = vstack([finalMatrix,neutralMatrix])
	# Shuffle is better for traing
	finalMatrix = shuffle(finalMatrix)
	train_Y = finalMatrix[:,-1].toarray()[:,0]
	return finalMatrix[:,:-1], train_Y, pMatrix, nMatrix, neMatrix

def classifyTweet(tweet, positiveMatrix, negativeMatrix, neutralMatrix):
	outputLabel = calssifyBaysian(tweet, positiveMatrix, negativeMatrix, neutralMatrix)
	print "According to Baysean Decision, the given tweet is ", outputLabel

def calssifyBaysian(tweet, positiveMatrix, negativeMatrix, neutralMatrix):

	# First use the ark tokenizer
	tokenisedTweet = tokenizeRawTweetText(tweet)
	tweet = ' '.join(tokenisedTweet)
	processedTweet, featureVector = processTweet(tweet)

	totalTokens = len(processedTweet)
	global wordnetDictionary, positiveCount, negativeCount, neutralCount, positiveProbability, \
			negativeProbability, neutralProbability, wordDictionary

	positiveFrequency = [1 for i in range(totalTokens)]
	negativeFrequency = [1 for i in range(totalTokens)]
	neutralFrequency = [1 for i in range(totalTokens)]

	wordnetScore = 0
	for token in processedTweet:
		if token in wordnetDictionary:
			wordnetScore += wordnetDictionary[token]

	for i in range(positiveCount):
		for j in range(totalTokens):
			if processedTweet[j] in wordDictionary and positiveMatrix[i, wordDictionary[processedTweet[j]]] == 1:
				positiveFrequency[j] += 1
    
	for i in range(negativeCount):
		for j in range(totalTokens):
			if processedTweet[j] in wordDictionary and negativeMatrix[i, wordDictionary[processedTweet[j]]] == 1:
				negativeFrequency[j] += 1

	for i in range(neutralCount):
		for j in range(totalTokens):
			if processedTweet[j] in wordDictionary and neutralMatrix[i, wordDictionary[processedTweet[j]]] == 1:
				neutralFrequency[j] += 1
    
    # Inititial probability
	positiveUnigramProbability = 10
	for i in range(totalTokens):
		positiveUnigramProbability *= float(positiveFrequency[j]) / float(positiveCount + 1)

	negativeUnigramProbability = 10
	for i in range(totalTokens):
		negativeUnigramProbability *= float(negativeFrequency[j]) / float(negativeCount + 1)

	neutralUnigramProbability = 10
	for i in range(totalTokens):
		neutralUnigramProbability *= float(neutralFrequency[j]) / float(negativeCount + 1)

	# Bayes probability
	posScore = positiveUnigramProbability * positiveUnigramProbability
	negScore = negativeUnigramProbability * negativeUnigramProbability
	neuScore = neutralUnigramProbability * neutralUnigramProbability
	
	if wordnetScore > 1:
		posScore *= 1000
		negScore *= 10
		neuScore *= 100
	elif wordnetScore > 0.6:
		posScore *= 10000
		negScore *= 5000
		neuScore *= 8000
	elif wordnetScore < -1:
		posScore *= (-1 * 10)
		negScore *= (-1 * 1000)
		neuScore *= (-1 * 100)
	elif wordnetScore < -0.6:
		posScore *= (-1 * 5000)
		negScore *= (-1 * 10000)
		neuScore *= (-1 * 8000)

	print 'bayesean score','p: ',posScore,'n: ',negScore,'neut: ',neuScore
	if posScore >= negScore:
		if posScore >= neuScore:
			return 'POSITIVE'
		else:
			return 'NEUTRAL'
    	else:
			if negScore >= neuScore:
				return 'NEGATIVE'
			else:
				return 'NEUTRAL'

if __name__ == '__main__':

	init()