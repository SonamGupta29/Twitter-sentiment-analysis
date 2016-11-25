import re
import sys
import random
from itertools import izip
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
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
afinnDict = getAFINNDictionary()
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
# Accuracies dictionary
accuraciesOfModels = dict()

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
	wordDictionary['AFINN_SCORE'] = 0

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

	# Train all model
	modelDict = trainAll(train_X, train_Y)

	# Display the accuracies
	print accuraciesOfModels	

	# Take the input from the user
	while True:
		tweet = str(raw_input("Enter the tweet : "))
		modelSelection = int(raw_input("Enter the model to choose from \n1. Naive Bayes  \n2. SVM\n3. NN\n4. BernoulliNB\n5.GaussianNB\n6.Adaboost\n7.GradientBoostingClassifier\t\tOption : "))
		# Baysian model
		if modelSelection == 1:
			classifyTweetBaysian(tweet, positiveMatrix, negativeMatrix, neutralMatrix)
		# SVM Model
		elif modelSelection == 2:
			classifyTweet(modelDict['SVM'], tweet)
		elif modelSelection == 3:
			classifyTweet(modelDict['NN'], tweet)
		elif modelSelection == 4:
			classifyTweet(modelDict['BNB'], tweet)	
		elif modelSelection == 5:
			#classifyTweet(modelDict['GNB'], tweet)	
			pass
		elif modelSelection == 6:
			classifyTweet(modelDict['Adaboost'], tweet)
		elif modelSelection == 7:
			#classifyTweet(modelDict['GBModel'], tweet)			
			pass
		# Choose all models and display the result	
		else:
			classifyTweetBaysian(tweet, positiveMatrix, negativeMatrix, neutralMatrix)
			classifyTweet(modelDict['SVM'], tweet)
			classifyTweet(modelDict['NN'], tweet)
			classifyTweet(modelDict['BNB'], tweet)
			#classifyTweet(modelDict['GNB'], tweet)
			classifyTweet(modelDict['Adaboost'], tweet)


def trainAll(train_X, train_Y):

	modelDict = dict()

	print "[INFO] Training SVM...",
	# Train SVM
	svmModel = OneVsOneClassifier(svm.LinearSVC(random_state = 0))
	accuraciesOfModels['SVM'] = cross_validation.cross_val_score(svmModel, train_X, train_Y, cv = 5).mean()	
	svmModel.fit(train_X,train_Y)	
	modelDict['SVM'] = svmModel
	print "[Completed]"
	
	# Train NN
	'''
		'lbfgs' ---> faster
		'sgd' refers to stochastic gradient descent.
		'adam' refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
	'''

	print "[INFO] Training Neural Network(adam)...",
	nnModel = OneVsOneClassifier(MLPClassifier(solver='adam', alpha = 1e-5, random_state = 0))
	accuraciesOfModels['NN'] = cross_validation.cross_val_score(nnModel, train_X, train_Y, cv = 5).mean()
	nnModel.fit(train_X, train_Y)
	modelDict['NN'] = nnModel
	print "[Completed]"
	
	
	print "[INFO] Training Bernoulli Naive Bayes...",
	# Multinomial Naive bayes 
	# NOT SO GOOD RESULT :(
	bnbModel = OneVsOneClassifier(BernoulliNB())
	bnbModel.fit(train_X, train_Y)
	accuraciesOfModels['BNB'] = cross_validation.cross_val_score(bnbModel, train_X, train_Y, cv = 5).mean()	
	modelDict['BNB'] = bnbModel
	print "[Completed]"
	

	'''
	# Gaussin NB
	# Required dense data 
	gnbModel = OneVsOneClassifier(GaussianNB())
	gnbModel.fit(train_X.toarray(), train_Y.toarray())
	accuraciesOfModels['GNB'] = cross_validation.cross_val_score(gnbModel, train_X.toarray(), train_Y.toarray(), cv = 5).mean()	
	modelDict['GNB'] = gbModel
	'''

	
	print "[INFO] Training Adaboost...",
	# Adaboost
	adaboostModel = OneVsOneClassifier(AdaBoostClassifier(n_estimators = 300))
	adaboostModel.fit(train_X, train_Y)
	accuraciesOfModels['Adaboost'] = cross_validation.cross_val_score(adaboostModel, train_X, train_Y, cv = 5).mean()
	modelDict['Adaboost'] = adaboostModel
	print "[Completed]"
		

	'''
	# Gradient boosting
	# Required dense data 
	gbModel = OneVsOneClassifier(GradientBoostingClassifier(n_estimators=300, learning_rate=1.0,max_depth=10, random_state=0))
	gbModel.fit(train_X,train_Y)
	accuraciesOfModels['Gradient Boosting'] = cross_validation.cross_val_score(gbModel, train_X, train_Y, cv = 5).mean()
	modelDict['GBModel'] = gbModel
	'''

	return modelDict


def classifyTweet(model, tweetText):

	global wordDictionary, acronymDict, emoticonDict, stopWordsDict, wordList, wordnetDictionary

	tokens = tokenizeRawTweetText(tweetText)
	processedTweet, featureDict = processTweet(' '.join(tokens))

	featureList = featureDict.values()
	features = dok_matrix((1,len(wordList)))
	# Add one form every word in the matrxi
	for word in processedTweet:
		try:
			features[0, wordDictionary[word]] = 1
		except Exception, e:
			pass

	features[0,wordDictionary['NON_ENG']] = featureList[0]
	features[0,wordDictionary['REPEAT']] = featureList[1]
	features[0,wordDictionary['EMOTICON']] = featureList[2]
	features[0,wordDictionary['ACRONYM']] = featureList[3]
	features[0,wordDictionary['WN_SCORE']] = featureList[4]
	features[0,wordDictionary['NN']] = featureList[5]
	features[0,wordDictionary['IN']] = featureList[6]
	features[0,wordDictionary['JJ']] = featureList[7]
	features[0,wordDictionary['AFINN_SCORE']] = featureList[8]
	features.tocsr()	

	senti = model.predict(features)
	senti = senti[0]
	if senti == 0:
		senti = 'positive'
	elif senti == 1:
		senti = 'negative'
	else:
		senti = 'neutral'

	print "tweet is : ", senti


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
				positiveMatrix[pos,wordDictionary['AFINN_SCORE']] = featureList[8]				
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
				negativeMatrix[neg,wordDictionary['AFINN_SCORE']] = featureList[8]

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
				neutralMatrix[neu,wordDictionary['AFINN_SCORE']] = featureList[8]				
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

def classifyTweetBaysian(tweet, positiveMatrix, negativeMatrix, neutralMatrix):
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