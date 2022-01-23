from random import randint, sample, shuffle
from ID3 import ID3

class Random_Forest():

    def __init__(self):
        self.vectors = []
        self.keys = []
        self.trees = []

    def train(self, trainingVectorsPath): 
        
        # flushing data structures
        self.vectors = []
        self.keys = []
        self.trees = []
        
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
            self.trees.append(ID3(tree_keys, tree_vecs))#create an ID3 object and add it to the forest
            self.trees[-1].buildTree(preset, stop_threshold)#build it's desision tree with the vectors created
            print("Tree #{} successfully built.".format(num+1))


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

    def __str__(self):
        return "randfor"