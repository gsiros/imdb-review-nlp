import os
from ID3 import ID3
from Bayes import NaiveBayesClassifier

def test(classifyfunc, pospath, negpath, limiter):
    i = 0
    counter = 0 
    correct = 0
    for filename in os.listdir(negpath):
        if i < limiter:
            path = os.path.join(negpath, filename)
            print(path)
            res = classifyfunc(path)
            if res == False:
                correct += 1
            counter += 1
            i += 1
        else: 
            break
    i = 0       
    for filename in os.listdir(pospath):
        if i < limiter:
            path = os.path.join(pospath, filename)
            print(path)
            res = classifyfunc(path)
            if res == True:
                correct += 1
            counter += 1
            i += 1
        else:
            break
        
    print("--- STATS ---")
    print("Accuracy: ", (correct/counter)*100, "%")

"""id3 = ID3.ID3()
id3.train("vectors/vectors_keys100_100.txt")
id3.buildTree()
test(id3.classify, "aclImdb/test/pos", "aclImdb/test/neg",12500)"""
nbc = NaiveBayesClassifier.NaiveBayesClassifier()
nbc.train("vectors/vectors_keys100_100.txt")
test(nbc.classify, "aclImdb/test/pos", "aclImdb/test/neg",1000)

