from collections import defaultdict

specialChar='1234567890#@%^&()_=`{}:"|[]\;\',./\n\t\r '

#Create wordnet dictionary
def getWordnetDictionary():
    wordNetDict = defaultdict()
    with open("wordnet.txt", 'r') as f:
        for line in f:
            line = line.rstrip()
            fields = line.split(":")
            wordNetDict[fields[0]] = float(fields[1])
    return wordNetDict

# Create en emoticon dictionary
def getEmoticonDictionary():
    emoticonDict = defaultdict()
    with open("emoticonsWithPolarity.txt",'r') as f:
        for line in f:
            line = line.strip('\n')
            fields = line.split(' ')            
            value = fields[-1]
            keys = fields[:-1]
            for key in keys:
                emoticonDict[key] = value
    return emoticonDict

# Create Acronym dict
def getAcronymDictionary():
    acronymDict = defaultdict()
    with open("acronym.txt",'r') as f:
        for line in f:
            line = line.rstrip('\n')
            key, value = line.split('\t')
            acronymDict[key] = value
    return acronymDict

# Create stopwords dict
def getStopwordDictionary():
    stopWordsDict = defaultdict(int)
    with open("stopWords.txt", "r") as f:
        for line in f:
            stopWordsDict[line.strip(specialChar).lower()] = 1
    return stopWordsDict

def getAFINNDictionary():
    afinnDict = defaultdict(int)
    with open("AFINN-111.txt", "r") as f:
        for line in f:
            key, value = line.split('\t')
            afinnDict[key] = int(value)
    return afinnDict
