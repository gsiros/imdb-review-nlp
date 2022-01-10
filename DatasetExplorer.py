import math
import os

class DatasetExplorer:

    def __init__(self, MAXNUM, percentage=1):
        self.attr = {}
        self.gains = {}
        self.percentage = percentage
        self.MAXNUM = MAXNUM

    def loadExamples(self, pospath, negpath):
        numOfExamples = int(self.MAXNUM * self.percentage)

        limiter = 0
        for filename in os.listdir(pospath):
            if limiter < numOfExamples:
                with open(os.path.join(pospath, filename), 'r') as f:
                    text = f.read()
                    words = text.split(" ")
                    alreadyChecked = []
                    for word in words:
                        cleanWord = word.strip(".,!").upper()
                        if cleanWord not in alreadyChecked:
                            if cleanWord in self.attr.keys():
                                self.attr[cleanWord][1] += 1
                            else:
                                self.attr[cleanWord] = [0,1]
                            alreadyChecked.append(cleanWord)
                    
            limiter += 1

        limiter = 0
        for filename in os.listdir(negpath):
            if limiter < numOfExamples:
                with open(os.path.join(negpath, filename), 'r') as f:
                    text = f.read()
                    words = text.split(" ")
                    alreadyChecked = []
                    for word in words:
                        cleanWord = word.strip(".,!").upper()
                        if cleanWord not in alreadyChecked:
                            if cleanWord in self.attr.keys():
                                self.attr[cleanWord][0] += 1
                            else:
                                self.attr[cleanWord] = [1,0]
                            alreadyChecked.append(cleanWord)
            limiter += 1

    def filterAttr(self, M):
        self.attr = dict(sorted(self.attr.items(),key = lambda x : x[1], reverse=True))
        i = 0
        for entry in self.attr.keys():
            if i < M: 
                print(entry, self.attr[entry][1], self.attr[entry][0], sep=" ")
                i+=1

    def __calcInfoGain(self, M):
        allExamples = int(self.MAXNUM * self.percentage)*2
        hc = self.__binEntropy(0.5)

        for word in self.attr.keys():
            #P(X=x)
            prob_of_word = (self.attr[word][0] + self.attr[word][1]) / (allExamples)
            #P(C=1|X=1)
            pC1X1 = 0
            if self.attr[word][0] + self.attr[word][1] != 0:
                pC1X1 = float((self.attr[word][1]) / (self.attr[word][0] + self.attr[word][1]))
            #P(C=1|X=0)
            pC1X0 = 0
            if self.attr[word][0] + self.attr[word][1] != allExamples:
                pC1X0 = float((allExamples/2 - self.attr[word][0]) / (allExamples - (self.attr[word][0] + self.attr[word][1])))
            #Entropies
            hcX1 = self.__binEntropy(pC1X1)
            hcX0 = self.__binEntropy(pC1X0)

            self.gains[word] = hc - ((prob_of_word*hcX1) + ((1-prob_of_word)*hcX0))

        self.gains = dict(sorted(self.gains.items(), key= lambda x: x[1], reverse=True))
        

    def createKeys(self, M):
        self.__calcInfoGain(M)
        filename = "keys{}_{}.txt".format(M, self.percentage*100)
        with open("keys/"+filename, "w") as outf:
            i,limit = 0,M
            for word in self.gains.keys():
                if i < limit:
                    if i != limit -1:
                        outf.write(word+"\n")
                    else:
                        outf.write(word)
                    print(word, self.gains[word], sep = " ")
                    i+=1
                else:
                    break


    def transformData(self, keypath, pospath, negpath, perc=1):
        numOfExamples = int(self.MAXNUM*perc)
        # LOAD KEYS
        keys = []
        with open(keypath, "r") as keyfile:
            keys.extend(keyfile.read().split("\n"))

        # Create the vectors...
        vectors = []
        
        # Negatives...
        limiter = 0
        for filename in os.listdir(negpath):
            if limiter < numOfExamples:
                with open(os.path.join(negpath, filename), 'r') as f:
                    print(os.path.join(pospath, filename))

                    text = f.read()
                    words = text.split(" ")
                    # Create 0-1 vector for each review:
                    # There will be as many 0/1s as the number
                    # of the keywords PLUS one more for the 
                    # #review (negative/positive).
                    vector = [0 for _ in range(len(keys)+1)]
                    for word in words:
                        cleanWord = word.strip(".,!").upper()
                        if cleanWord in keys:
                            vector[keys.index(cleanWord)] = 1
                    vectors.append(vector)
            limiter += 1

        # Positives...
        limiter = 0
        for filename in os.listdir(pospath):
            if limiter < numOfExamples:
                with open(os.path.join(pospath, filename), 'r') as f:
                    print(os.path.join(pospath, filename))
                    text = f.read()
                    words = text.split(" ")
                    # Create 0-1 vector for each review:
                    # There will be as many 0/1s as the number
                    # of the keywords PLUS one more for the 
                    # #review (negative/positive).
                    vector = [0 for _ in range(len(keys)+1)]
                    vector[-1] = 1
                    for word in words:
                        cleanWord = word.strip(".,!").upper()
                        if cleanWord in keys:
                            vector[keys.index(cleanWord)] = 1
                    vectors.append(vector)
            limiter += 1
        
        vectorfilename = "vectors/vectors_keys{}_{}.txt".format(len(keys), perc*100)
        with open(vectorfilename, "w") as vectorfile:
            header = ""
            for key in keys:
                header += key+","
            header += "RESULT"
            vectorfile.write(header+"\n")
            for vec in vectors:
                vectorfile.write(str(vec).strip("][")+"\n")

    def __binEntropy(self, prob):
        if prob == 0 or prob == 1:
            return 0
        else: 
            return - (prob * math.log2(prob)) - ((1-prob)*math.log2(1-prob))

de = DatasetExplorer(12500)
#de.loadExamples("aclImdb/train/pos", "aclImdb/train/neg")
#de.createKeys(30)
de.transformData("keys/keys30_100.txt", "aclImdb/train/pos", "aclImdb/train/neg", 0.8)