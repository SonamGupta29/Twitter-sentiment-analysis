import re
import sys
import random
from processTweets import *
from sklearn.multiclass import OneVsOneClassifier
from sklearn import svm
from sklearn import cross_validation
from sklearn.naive_bayes import BernoulliNB as NB
from scipy.sparse import *
import numpy as np
from sklearn.utils import shuffle
from ark import tokenizeRawTweetText
from createDictionaries import *
fp = open("processedTweets.txt", 'r')
#from flask import Flask,render_template, request, url_for

#app = Flask(__name__)
#@app.route('/')
def default():
	return render_template('tweet_submit.html')
	
#@app.route('/result', methods = ['POST'])
def result():
	tweet = request.form['yourtweet']
	senti = test_tweet_svm(tweet)
	return render_template('tweet_action.html',tweet=senti)

"""fp_tr=open("trainpreprocessedTweets.txt", 'w+')
fp_te=open("testpreprocessedTweets.txt", 'w+')
line=fp.readline()

ttc=0
while line:
    rnd=random.random()
    if rnd<0.99 :
        fp_te.write(line)
        ttc+=1
    else:
        fp_tr.write(line)
    line=fp.readline()

print "Total TestTweets %d" %(ttc)

fp_tr.close()
fp_te.close()
"""
fp = open("processedTweets.txt", 'r')

#Create dictionary of words

wordict={}

ed = getEmoticonDictionary()
ad = getAcronymDictionary()
swd = getStopwordDictionary()

pos_count=0
wordlist = []
neg_count=0
neu_count=0

def create_dict(file_to_read):
	global wordict,wordlist,pos_count,neg_count,neu_count
	fp = open(file_to_read,'r')
	line=fp.readline()
	while line:
	    line=line.rstrip()
	    fields=re.split(r'\t+',line)
	    if len(fields) < 2:
	        line=fp.readline()
	        continue
	    if "positive" == fields[0]:
	        pos_count+=1
	    elif "negative" == fields[0]:
	        neg_count+=1
	    else:
	        neu_count+=1
        
	    tokens=re.split(' ',fields[1])
	    size=len(tokens)
	    for i in range(size):
	        wordict[tokens[i]]=0
	    line=fp.readline()




#Read Features(NON_ENG,REPEAT,EMOTICON,ACRONYM and WN_SCORE) here and them to dictionary)
	wordict['NON_ENG']=0
	wordict['REPEAT']=0
	wordict['EMOTICON']=0
	wordict['ACRONYM']=0
	wordict['WN_SCORE']=0
	wordict['N_N']=0
	wordict['I_N']=0
	wordict['J_J']=0


#sort dictionary
	wordlist=sorted(wordict)

#Assign index number to each word in dictionary
	wordcount=0
	for word in wordlist:
	    wordict[word]=wordcount;
	    wordcount+=1
	
	fp.close()

create_dict('processedTweets.txt')
#print pos_count+neg_count+neu_count
#create boolean matrix (no. of tweets)*(no. of words in dict)

#pos_matrix = [[0 for i in range(wordcount)] for j in range(pos_count)]
#neg_matrix = [[0 for i in range(wordcount)] for j in range(neg_count)]
#neu_matrix = [[0 for i in range(wordcount)] for j in range(neu_count)]

def make_feature_bulk(pos_count,neg_count,neu_count,wordcount):
	global wordict
	print 'len',wordcount
	pos_matrix = dok_matrix((pos_count,wordcount))
	neg_matrix = dok_matrix((neg_count,wordcount))
	neu_matrix = dok_matrix((neu_count,wordcount))

	fp = open("processedTweets.txt", 'r')
	feature_fp=open("featureVectors.txt",'r')


	line=fp.readline()
	feature=feature_fp.readline()

	pos=0
	neg=0
	neu=0

	while line:
		line=line.rstrip()
		feature.rstrip()
		fields=re.split(r'\t+',line)
		featurelist=feature.split(" ")
    
		if len(fields) <2:
			feature=feature_fp.readline()
			line=fp.readline()
			continue
    	
		tokens=re.split(' ',fields[1])
    
		size=len(tokens)
		feature_size=len(featurelist)
    
		if "positive"==fields[0]:
			for i in range(size):
				pos_matrix[pos,wordict[tokens[i]]]=1
			pos_matrix[pos,wordict['NON_ENG']]=featurelist[0]
			pos_matrix[pos,wordict['REPEAT']]=featurelist[1]
			pos_matrix[pos,wordict['EMOTICON']]=featurelist[2]
			pos_matrix[pos,wordict['ACRONYM']]=featurelist[3]
			pos_matrix[pos,wordict['WN_SCORE']]=featurelist[4]
			pos_matrix[pos,wordict['N_N']]=featurelist[5]
        		pos_matrix[pos,wordict['I_N']]=featurelist[6]
   	    		pos_matrix[pos,wordict['J_J']]=featurelist[7]
			pos+=1

		elif "negative"==fields[0]:
			for i in range(size):
				neg_matrix[neg,wordict[tokens[i]]]=1
			neg_matrix[neg,wordict['NON_ENG']]=featurelist[0]
			neg_matrix[neg,wordict['REPEAT']]=featurelist[1]
			neg_matrix[neg,wordict['EMOTICON']]=featurelist[2]
			neg_matrix[neg,wordict['ACRONYM']]=featurelist[3]
			neg_matrix[neg,wordict['WN_SCORE']]=featurelist[4]
           		neg_matrix[neg,wordict['N_N']]=featurelist[5]
       			neg_matrix[neg,wordict['I_N']]=featurelist[6]
			neg_matrix[neg,wordict['J_J']]=featurelist[7]
			neg+=1

		else:
			for i in range(size):
				neu_matrix[neu,wordict[tokens[i]]]=1
			neu_matrix[neu,wordict['NON_ENG']]=featurelist[0]
			neu_matrix[neu,wordict['REPEAT']]=featurelist[1]
			neu_matrix[neu,wordict['EMOTICON']]=featurelist[2]
			neu_matrix[neu,wordict['ACRONYM']]=featurelist[3]
			neu_matrix[neu,wordict['WN_SCORE']]=featurelist[4]
           		neu_matrix[neu,wordict['N_N']]=featurelist[5]
	       		neu_matrix[neu,wordict['I_N']]=featurelist[6]
	       		neu_matrix[neu,wordict['J_J']]=featurelist[7]
			neu+=1

		line=fp.readline()
		feature=feature_fp.readline()
	p_matrix = pos_matrix
	n_matrix = neg_matrix
	ne_matrix = neu_matrix
	pos_matrix.tocsr()
	neg_matrix.tocsr()
	neu_matrix.tocsr()

	pos_matrix = hstack([pos_matrix,csr_matrix([[0],]*pos_count)])
	neg_matrix = hstack([neg_matrix,csr_matrix([[1],]*neg_count)])
	neu_matrix = hstack([neu_matrix,csr_matrix([[2],]*neu_count)])
	final_matrix = vstack([pos_matrix,neg_matrix])
	final_matrix = vstack([final_matrix,neu_matrix])
	final_matrix = shuffle(final_matrix)
	train_Y = final_matrix[:,-1].toarray()[:,0]
	return final_matrix[:,:-1],train_Y,p_matrix,n_matrix,ne_matrix

train_X,train_Y,pos_matrix,neg_matrix,neu_matrix= make_feature_bulk(pos_count,neg_count,neu_count,len(wordlist))
total_tweets=pos_count+neg_count+neu_count

def scoring(train_X,train_Y):
	score = cross_validation.cross_val_score(OneVsOneClassifier(NB()),train_X,train_Y,cv=5)
	#score = cross_validation.cross_val_score(OneVsOneClassifier(svm.LinearSVC(random_state=0)),train_X,train_Y,cv=5)
	print score
	print "average accuracy of svm ",score.mean()
#classifying

scoring(train_X,train_Y)

def svm_training(train_X,train_Y,kernel):
	if kernel == False:
		clf = OneVsOneClassifier(svm.LinearSVC(random_state=0))
	else:
		clf = OneVsOneClassifier(svm.SVC(kernel='rbf'))
	clf.fit(train_X,train_Y)
	return clf

clf = svm_training(train_X,train_Y,False)
    
pos_prob=float(pos_count)/float(total_tweets)
neg_prob=float(neg_count)/float(total_tweets)
neu_prob=float(neu_count)/float(total_tweets)

wn_dict = getWordnetDictionary()

def test_tweet_svm(tweet):
	global clf,worddict,ed,ad,swd,wordlist,wn_dict
	tweet,featurelist = processTweet(tweet)
	featurelist = featurelist.values()
	tweet = " ".join(tweet)
	tweet = tokenizeRawTweetText(tweet)
	tweet_size = len(tweet)
	#features = [0]*len(wordlist)
	features= dok_matrix((1,len(wordlist)))
	for i in range(tweet_size):
		if tweet[i] in wordict:
			features[0,wordict[tweet[i]]]=1
	features[0,wordict['NON_ENG']]=featurelist[0]
	features[0,wordict['REPEAT']]=featurelist[1]
	features[0,wordict['EMOTICON']]=featurelist[2]
	features[0,wordict['ACRONYM']]=featurelist[3]
	wn_score = 0
	for j in range(tweet_size):
		if tweet[j] in wn_dict:
			wn_score+=wn_dict[tweet[j]]
	features[0,wordict['WN_SCORE']]=str(wn_score)
	features.tocsr()
	senti = clf.predict(features)
	print senti
	senti = senti[0]
	if senti == 0:
		senti = 'positive'
	elif senti == 1:
		senti = 'negative'
	else:
		senti = 'neutral'
	return senti

	
def test_tweet_svm_bulk(file_to_read):
	fp = open(file_to_read,'r')
	output = open('test_result','w+')
	for line in fp:
		senti = test_tweet_svm(line)
		output.write(senti+'\n')
	output.close()
	
	


def user_interface(params):
	print "welcome system is trained and ready"
	if int(params[1]) == 0:
		test_tweet_svm_bulk(params[2])
		print "your output is present in file 'test_result'"
	elif int(params[1]) == 1:
		print "Enter Tweet"
		tweet = raw_input()
		senti1 = test_tweet_svm(tweet)
		print "According to svm",senti1
		senti2 = test_tweet_baysean(tweet)
		print "According to Baysean Decision function",senti2
		#user_interface(params)
	#else:
	#	global app
	#	app.run()

def test_tweet_baysean(test_tweet):
	test_tweet,features = processTweet(test_tweet)
	test_tweet=" ".join(test_tweet)
	ark_tokenised=tokenizeRawTweetText(test_tweet)
	tweet_size=len(ark_tokenised)
	global pos_matrix,neg_matrix,neu_matrix

	pos_tfreq=[1 for i in range(tweet_size)]
	neg_tfreq=[1 for i in range(tweet_size)]
	neu_tfreq=[1 for i in range(tweet_size)]

	wn_score=0
	for j in range(tweet_size):
		if ark_tokenised[j] in wn_dict:
                	wn_score+=wn_dict[ark_tokenised[j]]
                
    
	for i in range(pos_count):
		for j in range(tweet_size):
			if ark_tokenised[j] in wordict and pos_matrix[i,wordict[ark_tokenised[j]]]==1:
				pos_tfreq[j]+=1
    
    #print pos_tfreq
	for i in range(neg_count):
		for j in range(tweet_size):
			if ark_tokenised[j] in wordict and neg_matrix[i,wordict[ark_tokenised[j]]]==1:
				neg_tfreq[j]+=1

    
    
    #print neg_tfreq
    
	for i in range(neu_count):
		for j in range(tweet_size):
			if ark_tokenised[j] in wordict and neu_matrix[i,wordict[ark_tokenised[j]]]==1:
				neu_tfreq[j]+=1

    #print neu_tfreq
    
	pos_uni_prob=10
	for i in range(tweet_size):
		pos_uni_prob*=float(pos_tfreq[j])/float(pos_count+1)


	neg_uni_prob=10
	for i in range(tweet_size):
		neg_uni_prob*=float(neg_tfreq[j])/float(neg_count+1)

	neu_uni_prob=10
	for i in range(tweet_size):
		neu_uni_prob*=float(neu_tfreq[j])/float(neu_count+1)

	pos_given_tweet=pos_prob*pos_uni_prob
	neg_given_tweet=neg_prob*neg_uni_prob
	neu_given_tweet=neu_prob*neu_uni_prob
	
	if wn_score >1:
		pos_given_tweet*=1000
		neg_given_tweet*=10
		neu_given_tweet*=100
	elif wn_score >0.6:
		pos_given_tweet*=10000
		neg_given_tweet*=5000
		neu_given_tweet*=8000
	elif wn_score < -1:
		pos_given_tweet*=-1*10
		neg_given_tweet*=-1*1000
		neu_given_tweet*=-1*100
	elif wn_score<-0.6:
		pos_given_tweet*=-1*5000
		neg_given_tweet*=-1*10000
		neu_given_tweet*=-1*8000

	print 'bayesean score','p:',pos_given_tweet,'n:',neg_given_tweet,'neut:',neu_given_tweet
	if pos_given_tweet>=neg_given_tweet:
		if pos_given_tweet>=neu_given_tweet:
			return 'positive'
		else:
			return 'neutral'
    	else:
		if neg_given_tweet>=neu_given_tweet:
			return 'negative'
		else:
			return 'neutral'


if __name__ == '__main__':
	user_interface(sys.argv)

