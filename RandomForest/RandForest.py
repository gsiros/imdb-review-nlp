from random import randint
from ID3 import ID3

class Random_Forest:

    def __init__(self):
        self.vectors = []
        self.keys = []
        self.trees = []

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

    def random_forest(self, trainingVectors, keys, m, forest_size, preset, stop_threshold):
        #populate forest
        for i in range(forest_size):
            tree_vecs = []
            tree_keys = []
            for j in range(len(trainingVectors)): #choose a random equal sized subset of the vectors from the trainning data
                tree_vecs.append(trainingVectors[randint(len(trainingVectors))])
            temp_keys = keys.copy()#choose a random m-sized subset of the keys
            for j in range(m):
                tree_keys.append[temp_keys.pop(randint(len(temp_keys)))]
            self.trees.append(ID3.ID3())#create an ID3 object and add it to the forest
            self.trees[j].buildTree(tree_vecs, tree_keys, preset, stop_threshold)#build it's desision tree with the vectors created


    def classify(self, revpath):
        #classify in each tree
        results = []
        for item in self.trees: 
            results.append(item.classify(revpath))

        #count votes
        counter_pos = 0 
        counter_neg = 0
        for item in results:
            if item == True:
                counter_pos += 1
            else:
                counter_neg += 1
                
        return counter_pos > counter_neg
