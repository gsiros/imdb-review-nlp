import os

class NaiveBayesClassifier():
    
    def __init__(self):
        self.keys = []
        self.vectors = []
        self.hc = 0

    def train(self, trainingVectorsPath):

        # flush data structures...
        self.keys = []
        self.vectors = []

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
        # Vector-ify the review text...
        #print(revpath)
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
        #print(rev_vector)
        # if P(C = 1 | X) > P(C = 0 | X)
        #   RETURN 1

        #P(C=1|X):
        #P(X=xi | C=1):
        # Σε πόσα μηνύματα εκπαίδευσης της κατηγορίας c=1
        # εμφανίζεται η λέξη που αντιστοιχεί στη Xi ;
        attr_counter_pos = [0 for _ in range(len(self.keys))]
        attr_counter_neg = [0 for _ in range(len(self.keys))]

        # for every training vector
        for vector in self.vectors:
            # check every key attribute
            for i in range(len(self.keys)):
                # if review vector is positive:
                if vector[-1] == 1:
                    # if the vector has a key attribute
                    if vector[i] == rev_vector[i]:
                        attr_counter_pos[i] += 1
                else:
                    if vector[i] == rev_vector[i]:
                        attr_counter_neg[i] += 1
                            

       
        pc1x = (1/2)
        for counter in attr_counter_pos:
            if counter != 0:
                pc1x *= counter / (len(self.vectors ) // 2)
            else:
                pc1x *= 1 / ((len(self.vectors) // 2) + 2)

        pc0x = (1/2)
        for counter in attr_counter_neg:
            if counter != 0:
                pc0x *= (counter / (len(self.vectors) // 2))
            else:
                pc0x *= (1 / ((len(self.vectors) // 2) + 2))

        if pc1x > pc0x:
            return True
        else:
            return False

    def __str__(self):
        return "nbc"