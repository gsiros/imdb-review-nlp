import os

def test(classifyfunc, pospath, negpath, limiter):
    i = 0
    counter = 0 
    correct = 0
    for filename in os.listdir(pospath):
        if i < limiter:
            path = os.path.join(pospath, filename)
            print(path)
            res = classifyfunc(path)
            if res == 1:
                correct += 1
            counter += 1
            i += 1
        else: 
            break
    i = 0       
    for filename in os.listdir(negpath):
        if i < limiter:
            path = os.path.join(negpath, filename)
            print(path)
            res = classifyfunc(path)
            if res == 0:
                correct += 1
            counter += 1
            i += 1
        else:
            break
        
    print("--- STATS ---")
    print("Accuracy: ", (correct/counter)*100, "%")

class NaiveBayesClassifier():
    
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
        # if P(C = 1 | X) > P(C = 0 | X)
        #   RETURN 1

        #P(C=1|X):
        #P(X=xi | C=1):
        # Σε πόσα μηνύματα εκπαίδευσης της κατηγορίας c=1
        # εμφανίζεται η λέξη που αντιστοιχεί στη Xi ;
        attr_counter_pos = [0 for _ in range(len(self.keys))]
        attr_counter_neg = [0 for _ in range(len(self.keys))]

        for vector in self.vectors:
            if vector[-1] == 1:
                for i in range(len(vector)-1):
                    if rev_vector[i] == 1:
                        attr_counter_pos[i] += vector[i]
            else:
                for i in range(len(vector)-1):
                    if rev_vector[i] == 1:
                        attr_counter_neg[i] += vector[i]

       
        pc1x = (1/2)
        for counter in attr_counter_pos:
            if counter != 0:
                pc1x *= counter / (len(self.vectors) // 2)
            else:
                pc1x *= 1 / ((len(self.vectors) // 2) + 2)

        pc0x = (1/2)
        for counter in attr_counter_neg:
            if counter != 0:
                pc0x *= counter / (len(self.vectors) // 2)
            else:
                pc0x *= 1 / ((len(self.vectors) // 2) + 2)

        #print(pc1x, pc0x)
        if pc1x > pc0x:
            return 1
        else:
            return 0

nbc = NaiveBayesClassifier()
nbc.train("vectors/vectors_keys100_100.txt")
#print(nbc.classify("aclImdb/test/pos/8_9.txt"))
test(nbc.classify, "aclImdb/test/pos", "aclImdb/test/neg", 200)
#print(nbc.classify("aclImdb/test/neg/12398_1.txt"))