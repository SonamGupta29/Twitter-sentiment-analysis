import re
import sys

from processTweets import *
from ark import *

# Get the corpus filename from the user for the preprocessing of the tweets

try:
	corpusFileName = sys.argv[1]
except:
	corpusFileName = raw_input("Enter the filename : ")

processedTweetFile = open('processedTweets.txt', 'w+')
featureVectorFile = open('featureVectors.txt', 'w+')

'''
	Read the line by line, 
	If we observe the format of the file is "polarity		Tweet Text"
	so read it line by line and seperate the polarity and tweet text and 
	preprocess the tweet text
'''

count = 0
with open(corpusFileName, 'r') as f:
    for line in f:
        count += 1
        print "[INFO] Processing ", count, "tweet"
    	fields = re.split(r'\t+', line)
    	polarity = fields[0].strip();
    	tweetText = fields[1].strip();
    	'''
    		Now as we have got the tweetText, we will preprocess it
    	'''
        tokens = tokenizeRawTweetText(tweetText)
        processedTweet, featureDict = processTweet(' '.join(tokens))

        # Save the preprocessed tweet along with the polarity
        featureVectorFile.write(' '.join(featureDict.values()) + '\n')
        processedTweetFile.write(polarity + '\t' + ' '.join(processedTweet) + '\n')

processedTweetFile.close()
featureVectorFile.close()