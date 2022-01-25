import math
import os

class DatasetExplorer:
    """Support class for keyword & training vector file creation.
    """

    def __init__(self, MAXNUM, percentage=1):
        """Constructor
        
        MAXNUM -- (int) the maximum number of training examples per category
        percentage -- (float) real number in [0,1] that works as a limit to how many training examples to load
        
        >>>de = DatasetExplorer(12500)"""
        self.attr = {}
        self.gains = {}
        self.percentage = percentage
        self.MAXNUM = MAXNUM

    def loadExamples(self, pospath, negpath):
        """Function that loads the training examples to the explorer instance.
        
        pospath -- (str) path to the positive review training folder
        negpath -- (str) path to the negative review training folder

        >>>de.loadExamples("some/path/pos", "some/path/neg")"""

        # Calculate the maximum number of examples to insert
        # from each of the two categories according to the
        # requested percentage.
        # 
        # ex. 12500 reviews in each of the two categories 
        # times 0.5 (percentage) equals 12500*0.5=6250 reviews
        # from each of the two categories.       
        numOfExamples = int(self.MAXNUM * self.percentage)

        # Load the positive reviews:

        limiter = 0
        # For each file in the positive review folder:
        for filename in os.listdir(pospath):
            if limiter < numOfExamples:
                # Open file:
                with open(os.path.join(pospath, filename), 'r', encoding='utf-8') as f:
                    # Read the review and split it to each
                    # individual word.
                    text = f.read()
                    words = text.split(" ")
                    alreadyChecked = []
                    # For every word in the text:
                    for word in words:
                        # Clean the word from punctuation signs
                        # and make it capital it uppercase.
                        cleanWord = word.strip(".,!").upper()
                        # If the word is not already checked
                        # in the same review text:
                        if cleanWord not in alreadyChecked:
                            # If the word has already an entry in 
                            # the keyword dictionary data structure:
                            if cleanWord in self.attr.keys():
                                # Increase its appearances in
                                # POSITIVE reviews by +1.
                                self.attr[cleanWord][1] += 1
                            else:
                                # Otherwise create a new entry.
                                self.attr[cleanWord] = [0,1]
                            # Include word in already checked words
                            # in the particular review.
                            alreadyChecked.append(cleanWord)
                    
            limiter += 1


        # The same as described above for the negative reviews:
        limiter = 0
        for filename in os.listdir(negpath):
            if limiter < numOfExamples:
                with open(os.path.join(negpath, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    words = text.split(" ")
                    alreadyChecked = []
                    for word in words:
                        cleanWord = word.strip(".,!").upper()
                        if cleanWord not in alreadyChecked:
                            if cleanWord in self.attr.keys():
                                # Increase its appearances in
                                # NEGATIVE reviews by +1.
                                self.attr[cleanWord][0] += 1
                            else:
                                # Otherwise create a new entry.
                                self.attr[cleanWord] = [1,0]
                            alreadyChecked.append(cleanWord)
            limiter += 1

    def __calcInfoGain(self):
        """(PRIVATE) method that calculates the information gain of each word that is found in the review texts.

        """
        # Calculate the *TOTAL* number of reviews in both categories:
        allExamples = int(self.MAXNUM * self.percentage)*2
        
        # Calculate the information gain of the positive/negative categories
        # (#positive_reviews == #negative_reviews)
        # *IMPORTANT*
        # The DatasetExplorer reads the same number of positive and negative
        # reviews for balance purposes.
        hc = self.__binEntropy(0.5)
        
        
        # For every word in the keyword attribute dictionary:
        for word in self.attr.keys():
            # P(X=x) = (#appearances in negative reviews + #appearances in positive reviews) / #total_reviews  
            prob_of_word = (self.attr[word][0] + self.attr[word][1]) / (allExamples)
            
            #P(C=1|X=1):
            pC1X1 = 0
            # If the word appears at least once in the reviews:
            if self.attr[word][0] + self.attr[word][1] != 0:
                #P(C=1|X=1) = (#appearances in positve reviews) / (#appearances in positve reviews + #appearances in negative reviews)
                pC1X1 = float((self.attr[word][1]) / (self.attr[word][0] + self.attr[word][1]))
            
            #P(C=1|X=0):
            pC1X0 = 0
            # If the word does not appear at least once in the reviews:
            if self.attr[word][0] + self.attr[word][1] != allExamples:
                #P(C=1|X=0) = (# positive reviews where the word does not appear) / (# reviews that the word does not appear at all)
                pC1X0 = float((allExamples/2 - self.attr[word][1]) / (allExamples - (self.attr[word][0] + self.attr[word][1])))

            #Entropies
            hcX1 = self.__binEntropy(pC1X1)
            hcX0 = self.__binEntropy(pC1X0)

            # Calculate Information gain...
            self.gains[word] = hc - ((prob_of_word*hcX1) + ((1-prob_of_word)*hcX0))

        # Sort the words according to their information gain in descending oreder
        self.gains = dict(sorted(self.gains.items(), key= lambda x: x[1], reverse=True))
        
        # Post-infogain filtering according to external stop-lists:
        commonwords = []
        with open("non_negative_or_positive_connotation_words.txt", 'r', encoding='utf-8') as f:
            text = f.read()
            words = text.split("\n")
            commonwords.extend([word.strip(".,!").upper() for word in words])
        for key in list(self.gains.keys()):
            if key.isnumeric() or "/" in key:
                del self.gains[key]
            elif key in commonwords:
                if self.gains[key] < 0.01:
                    del self.gains[key]
        

    def createKeys(self, M):
        """Method that creates a file with the top M keywords in
        descending order according to their information gain.
        
        M -- (int) the number of keywords to generate a file with"""
        
        # Calculate the information gain of each key:
        self.__calcInfoGain()

        filename = "keys{}_{}.txt".format(M, self.percentage)
        # Write the keywords to a file:
        with open("keys/TEST"+filename, "w", encoding='utf-8') as outf:
            i,limit = 0,M
            for word in self.gains.keys():
                if i < limit:
                    if i != limit -1:
                        outf.write(word+"\n")
                    else:
                        outf.write(word)
                    print(word, self.gains[word])
                    i+=1
                else:
                    break


    def transformData(self, keypath, pospath, negpath, perc=1):
        """Method that creates the training vector files from a keyword file.
        
        keypath -- (str) path to the keyword file
        pospath -- (str) path to the positive review training folder
        negpath -- (str) path to the negative review training folder"""
        
        # Calculate the number of reviews in each category:
        numOfExamples = int(self.MAXNUM*perc)
        
        # LOAD KEYS from keyword file:
        keys = []
        with open(keypath, "r") as keyfile:
            keys.extend(keyfile.read().split("\n"))

        # Create the training vectors...
        vectors = []
        
        # Transform negative reviews in training vectors of 0-1s:
        limiter = 0
        for filename in os.listdir(negpath):
            if limiter < numOfExamples:
                with open(os.path.join(negpath, filename), 'r', encoding='utf-8') as f:
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

        # Same for the positive reviews...
        limiter = 0
        for filename in os.listdir(pospath):
            if limiter < numOfExamples:
                with open(os.path.join(pospath, filename), 'r', encoding='utf-8') as f:
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
        
        # Create the vector file with the training vectors:
        vectorfilename = "vectors/vectors_keys{}_{}.txt".format(len(keys), float(perc))
        with open(vectorfilename, "w", encoding='utf-8') as vectorfile:
            header = ""
            for key in keys:
                header += key+","
            header += "RESULT"
            vectorfile.write(header+"\n")
            for vec in vectors:
                vectorfile.write(str(vec).strip("][")+"\n")

    def __binEntropy(self, prob):
        """(PRIVATE) support method that calculates the entropy of two categories.
        
        prob -- (float) the probability of one of the two categories"""
        if prob == 0 or prob == 1:
            return 0
        else: 
            return - (prob * math.log2(prob)) - ((1-prob)*math.log2(1-prob))
