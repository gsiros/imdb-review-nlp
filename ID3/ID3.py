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

    ## SETTERS ##
    def setParent(self, parentNode):
        self.parent = parentNode

    def setLeft(self, leftChildNode):
        self.left = leftChildNode

    def setRight(self, rightChildNode):
        self.right = rightChildNode

    def setAttribute(self, attribute):
        self.attribute = attribute

    def setNumberOfPositives(self, numOfPositives):
        self.numPos = numOfPositives
        self.sum = self.numPos + self.numNeg

    def setNumberOfNegatives(self, numOfNegatives):
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

    def __str__(self) -> str:
        return "[ \n\tAttribute:'{}',\n\tLeftChild:'{}',\n\tRightChild:'{}',\n\tNumOfPos:'{}',\n\tNumOfNeg:'{}',\n\tSum:'{}'\n]".format(
            self.attribute,
            self.left,
            self.right,
            self.numPos,
            self.numNeg,
            self.sum
        )
    
class ID3:

    def __init__(self):
        self.keys = []
        self.vectors = []

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
        pass

    def __buildTree(self, trainingVectors, keys, preset=1):
        """Builds the decision tree based on training vectors.
        
        trainingVectors -- (list) the training vectors; list of 0-1 lists
        keys -- (list) the key attributes
        preset -- the preset category for classification (1 POS - 0 NEG)
        """
        # build the tree...
        pass

    def __id3(self, trainingVectors, keys, preset=1):
        """(RECURSIVE USE) The main ID3 algorithm.
        
        trainingVectors -- (list) the training vectors; list of 0-1 lists
        keys -- (list) the key attributes
        preset -- the preset category for classification (1 POS - 0 NEG)
        """
        # to implement
        return None