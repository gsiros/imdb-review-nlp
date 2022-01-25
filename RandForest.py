from random import randint, sample, shuffle
from ID3 import ID3

class Random_Forest:
    """Random Forest Classifier class"""

    def __init__(self):
        self.vectors = []
        self.keys = []
        self.trees = []

    def train(self, trainingVectorsPath): 
        """Method that trains the algorithm from the training vector file provided.

        trainingVectorsPath -- (str) path to the training vector file.
        """

        # Flush data structures for training...
        self.vectors = []
        self.keys = []
        self.trees = []
        
        # Open the training vector file:
        with open(trainingVectorsPath, "r", encoding='utf-8') as trainingfile:
            lines = trainingfile.readlines()

            # Read the keyword line and extract the keywords:
            self.keys.extend(lines[0].split(","))
            self.keys.pop(-1)
            lines.pop(0)
            
            # Read the rest of the training vectors and parse them
            # to a list:
            for line in lines:
                vector = line.strip("\n").split(",")
                vector = [int(item) for item in vector]
                self.vectors.append(vector)
            # Shuffle the order of training vectors to increase
            # the randomness of the tree bags.
            shuffle(self.vectors)

    def random_forest(self, m, forest_size, preset, stop_threshold):
        tree_vecs = []
        #populate forest
        for num in range(forest_size):
            # pick random subset for the keys
            tree_keys = sample(self.keys, m)
            # find their indeces
            indeces = [self.keys.index(key) for key in tree_keys]
            for _ in range(len(self.vectors)): #choose a random equal sized subset of the vectors from the trainning data
                picked_vector = self.vectors[randint(0,len(self.vectors)-1)]
                # remove columns that correspond to non existent keys
                # the following part uses the indeces of the new keys from
                # the new subset to include only the 0/1 data that correspond to
                # them
                cleared_random_vector = [picked_vector[index] for index in indeces]
                # append result column for training (positive/negative)
                cleared_random_vector.append(picked_vector[-1])
                # add vector to tree vectors
                tree_vecs.append(cleared_random_vector)
            
            #create an ID3 object and add it to the forest    
            self.trees.append(ID3(tree_keys, tree_vecs))
            #build it's desision tree with the vectors created
            self.trees[-1].buildTree(preset, stop_threshold)
            print("Tree #{} successfully built.".format(num+1))


    def classify(self, revpath):
        # Call the classify method on each ID3 instance:
        results = []
        for item in self.trees: 
            results.append(item.classify(revpath))

        # Count votes:
        counter_pos = 0 
        counter_neg = 0
        for item in results:
            if item == True:
                counter_pos += 1
            else:
                counter_neg += 1
                
        return counter_pos > counter_neg

    def __str__(self):
        return "randfor"