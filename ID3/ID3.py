import math

class ID3Node:
    def __init__(self, parent=None, attribute=None):
        # Father of node
        self.parent = parent
        # Left child (Positive, 1)
        self.left = None
        # Right chid (Negative, 0)
        self.right = None
        # The attribute
        self.attribute = attribute
        # number of positives
        self.numPos = 0
        # number of negatives
        self.numNeg = 0
        # sum of positives and negatives
        self.sum = self.numPos + self.numNeg
        # preset value
        self.preset = None
        

    ## SETTERS ##
    def setPreset(self, preset):
        self.preset = preset

    def setParent(self, parentNode): #no need
        self.parent = parentNode

    def setLeft(self, leftChildNode):
        self.left = leftChildNode

    def setRight(self, rightChildNode):
        self.right = rightChildNode

    def setAttribute(self, attribute):
        self.attribute = attribute

    def setNumberOfPositives(self, numOfPositives): # too bad
        self.numPos = numOfPositives
        self.sum = self.numPos + self.numNeg

    def setNumberOfNegatives(self, numOfNegatives): # too bad
        self.numNeg = numOfNegatives
        self.sum = self.numPos + self.numNeg

    ## GETTERS ##
    def getParent(self):
        return self.parent

    def getLeft(self):
        return self.left

    def getRight(self):
        return self.right

    def getAttribute(self):
        return self.attribute

    def getNumberOfPositives(self):
        return self.numPos

    def getNumberOfNegatives(self):
        return self.numNeg
    
    def getSumOfPosNeg(self):
        return self.sum

    def getPreset(self):
        return self.preset

    def __str__(self) -> str:
        #return "NODE: {},\nLEFT: {},\nRIGHT: {}".format(self.attribute, self.left.getAttribute(), self.right.getAttribute())
        return str(self.attribute)
        #return "[ \n\tAttribute:'{}',\n\tLeftChild:'{}',\n\tRightChild:'{}',\n\tNumOfPos:'{}',\n\tNumOfNeg:'{}',\n\tSum:'{}'\n]".format(
            #self.attribute,
            #self.left,
            #self.right,
            #self.numPos,
            #self.numNeg,
            #self.sum
        #)
    
class ID3:

    
    def __init__(self):
        self.keys = []
        self.vectors = []
        self.gains = {}
        self.desision_tree = None

    


    #loads training data
    def train(self, trainingVectorsPath): 
        with open(trainingVectorsPath, "r") as trainingfile:
            lines = trainingfile.readlines()

            # get keys
            self.keys.extend(lines[0].split(","))
            self.keys.pop(-1)
            lines.pop(0)
            
            # get the training vectors
            for line in lines:
                vector = line.strip("\n").split(",")
                vector = [int(item) for item in vector]
                self.vectors.append(vector)


    def classify(self, revpath):
        """Classifies a review to POSITIVE or NEGATIVE.
        revpath -- the path of the review .txt file
        """
        # to implement
        rev_vector = [0 for _ in range(len(self.keys)+1)]
        with open(revpath, "r") as revfile:
            rev_text = revfile.read()
            #print(rev_text)
            words = rev_text.split(" ")
            for word in words:
                cleanWord = word.strip(".,!").upper()
                if cleanWord in self.keys:
                    #print(cleanWord)
                    rev_vector[self.keys.index(cleanWord)] = 1

        curr_node = self.desision_tree
        while curr_node != None: #We have not reched a leaf
            if curr_node.getAttribute() != None:
                if rev_vector[self.keys.index(curr_node.getAttribute())] == 1:
                    curr_node = curr_node.getRight()
                else:
                    curr_node = curr_node.getLeft()
            else:
                return curr_node.getPreset()
            


    def buildTree(self, preset = False, stop_threshold = 0.1):
        """Builds the decision tree based on training vectors.
        
        trainingVectors -- (list) the training vectors; list of 0-1 lists
        keys -- (list) the key attributes
        preset -- the preset category for classification (1 POS - 0 NEG)
        """
        # build the tree...
        self.desision_tree = self.__id3(self.vectors, self.keys, preset, stop_threshold)

        
        
        
    # calculate most valuable key and return it
    def __bestKey(self, vectors, keys):
        allExamples = len(vectors)
        hc = self.__binEntropy(0.5)
        gains = []

        for col in range(len(keys)):
            count_1_pos = 0 # reviews with attribute value of 1 that are possitive
            count_1_neg = 0 # reviews with attribute value of 1 that are negative
            count_0_pos = 0 # reviews with attribute value of 0 that are possitive
            count_0_neg = 0 # reviews with attribute value of 0 that are negative

            for vector in vectors:
                if vector[col] == 1:
                    if vector[-1] == 0:
                        count_1_pos+= 1
                    else:
                        count_1_neg+= 1
                else:
                    if vector[-1] == 0:
                        count_1_pos+= 1
                    else:
                        count_1_neg+= 1
                    
            #P(X=1)
            prob_of_word = (count_1_pos + count_1_neg) / (allExamples)
            #P(C=1|X=1)
            pC1X1 = 0
            if count_1_pos + count_1_neg != 0:
                pC1X1 = float((count_1_pos) / (count_1_pos + count_1_neg))
            #P(C=1|X=0)
            pC1X0 = 0
            if count_0_pos + count_0_neg != 0:
                pC1X0 = float((count_0_pos) / (count_0_pos + count_0_neg))
            #Entropies
            hcX1 = self.__binEntropy(pC1X1)
            hcX0 = self.__binEntropy(pC1X0)

            gains.append(hc - ((prob_of_word*hcX1) + ((1-prob_of_word)*hcX0)))

        max_gain = max(gains)
        return keys[gains.index(max_gain)] # to be used in buildTree
        
       

    def __binEntropy(self, prob):
        if prob == 0 or prob == 1:
            return 0
        else: 
            return - (prob * math.log2(prob)) - ((1-prob)*math.log2(1-prob))

    def __id3(self, trainingVectors, keys, preset, stop_threshold):
        """(RECURSIVE USE) The main ID3 algorithm.
        
        trainingVectors -- (list) the training vectors; list of 0-1 lists
        keys -- (list) the key attributes
        preset -- the preset category for classification (1 POS - 0 NEG)
        """
        # to implement
        node = ID3Node()
        if trainingVectors == []:
            node.setPreset(preset)
            return node

        count_pos = 0 
        count_neg = 0
        for item in trainingVectors:
            if item[-1] == 1:
                count_pos += 1
            else:
                count_neg += 1
       
        if keys == []:
            if count_pos >= count_neg:
                node.setPreset(True)
            else:
                node.setPreset(False)
            return node
    
        if (count_pos>0 and (count_neg/count_pos)<= stop_threshold): # 95% of training data are possitive. Stop to avoid overfitting
            node.setPreset(True)
            return node
        if (count_neg>0 and (count_pos/count_neg)<= stop_threshold): # 95% of training data are negative. Stop to avoid overfitting
            node.setPreset(False)
            return node

        best_key = self.__bestKey(trainingVectors,keys)
        best_index = keys.index(best_key) # the index of the best key in the key vector
        node.setAttribute(best_key)

        with_key = [] #list of reviews containing the key word
        without_key = [] #list of reviews *not* containing the key word
        for item in trainingVectors:
            if item[best_index] == 1:
                with_key.append(item[:best_index]+item[best_index+1:]) # the attribute selected in this step is no longer useful in the next steps
            else:
                without_key.append(item[:best_index]+item[best_index+1:])
        

        #since keys can either exist or not exist in a review, we only need to make 2 children
        #we pass the key list without the key we used in this step (best key)
        node.setLeft(self.__id3(without_key, keys[:best_index]+keys[best_index+1:], preset, stop_threshold))
        node.setRight(self.__id3(with_key, keys[:best_index]+keys[best_index+1:], preset, stop_threshold))
        return node

    def printTree(self):
        string = ""
        stack = []
        stack.append(self.desision_tree)
        while len(stack) != 0:
            print("heil")
            index = stack.pop()
            if index != None:
                if index.getAttribute() != None:
                    string += index.getAttribute() + "({})".format(str(index.getParent())) + "\n"
                else:
                    string += str(index.getPreset()) + "({})".format(str(index.getParent())) + "\n"
                
                if index.getLeft() != None:
                    index.getLeft().setParent(index)
                if index.getRight() != None:
                    index.getRight().setParent(index)
                stack.append(index.getLeft())
                stack.append(index.getRight())
        print(string) 