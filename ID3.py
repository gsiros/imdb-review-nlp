import math

class ID3Node:
    def __init__(self, parent=None, attribute=None):
        # Father of node
        self.parent = parent
        # Left child (DOES NOT have the current attribute-keyword)
        self.left = None
        # Right chid (DOES have the current attribute-keyword)
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
        return str(self.attribute)
        
    
class ID3:
    """ID3 Classifier class"""
    
    def __init__(self, keys = [], vectors = []):
        self.keys = keys
        self.vectors = vectors
        self.gains = {}
        self.desision_tree_root_root = None
        self.hc = 0

    def train(self, trainingVectorsPath): 
        """Method that trains the algorithm from the training vector file provided.

        trainingVectorsPath -- (str) path to the training vector file.
        """

        # Flush data structures for training...
        self.keys = []
        self.vectors = []
        self.gains = {}
        self.desision_tree_root_root = None

        # Open the training vector file:
        with open(trainingVectorsPath, "r") as trainingfile:
            lines = trainingfile.readlines()

            # Read the keyword line and extract the keywords:
            self.keys.extend(lines[0].split(","))
            self.keys.pop(-1)
            lines.pop(0)
            
            # Count how many vectors are positive
            count_pos = 0
            # Read the rest of the training vectors and parse them
            # to a list:
            for line in lines:
                vector = line.strip("\n").split(",")
                vector = [int(item) for item in vector]
                count_pos += vector[-1]
                self.vectors.append(vector)
            # Calculate the entropy for the positive category:
            self.hc = self.__binEntropy(count_pos/len(self.vectors))
            

    def classify(self, revpath):
        """Main classification method that classifies a review as positive (True) or negative (False)
        
        revpath - (str) path to the review file"""
        
        # Initialize a dummy training vector full of 0s.     
        rev_vector = [0 for _ in range(len(self.keys)+1)]

        # Open the review file:
        with open(revpath, "r") as revfile:
            rev_text = revfile.read()
            words = rev_text.split(" ")
            for word in words:
                cleanWord = word.strip(".,!").upper()
                if cleanWord in self.keys:
                    # Vector-ify the review text...
                    # ex. review_vector[indexOf('BAD')] = 1
                    rev_vector[self.keys.index(cleanWord)] = 1

        # Traverse the decision tree, starting from the root:
        curr_node = self.desision_tree_root
        while curr_node != None: # We have not reched a leaf
            
            if curr_node.getAttribute() != None: # We have not reached a preset-leaf
                # If node's keyword appears in the incoming to-be-classified
                # review vector, move to the right child node 
                if rev_vector[self.keys.index(curr_node.getAttribute())] == 1:
                    curr_node = curr_node.getRight()
                else:
                    # Else move to the left side:
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
        print("Building decision tree...")
        self.desision_tree_root_root = self.__id3(self.vectors, self.keys, preset, stop_threshold)

        
    # calculate most valuable key and return it
    def __bestKey(self, vectors, keys):
        allExamples = len(vectors)
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

            gains.append(self.hc - ((prob_of_word*hcX1) + ((1-prob_of_word)*hcX0)))

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

    def __str__(self):
        return "id3"