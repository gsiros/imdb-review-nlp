import os
from random import randint

from ID3 import ID3
from  NaiveBayesClassifier import NaiveBayesClassifier
from RandForest import Random_Forest
from DatasetExplorer import DatasetExplorer

class Tester:
    
    def __init__(self, classifier):
        self.classifier = classifier

    def buildTestVectorFiles(self, keyNum, trainpospath, trainnegpath):
        for percentage in range(2,50,2):
            print("Building training vector file with {} keys on {}% of training data...".format(keyNum, percentage/1000))
            de = DatasetExplorer(12500,percentage/1000)
            de.loadExamples(trainpospath, trainnegpath)
            de.createKeys(keyNum)
            print("Building...")
            de.transformData(
                "keys/TESTkeys{}_{}.txt".format(keyNum, percentage/1000),
                trainpospath,
                trainnegpath,
                percentage/1000
            )
            print("Done!")
        
        for percentage in range(5,101,5):
            print("Building training vector file with {} keys on {}% of training data...".format(keyNum, percentage))
            de = DatasetExplorer(12500,percentage/100)
            de.loadExamples(trainpospath, trainnegpath)
            de.createKeys(keyNum)
            print("Building...")
            de.transformData(
                "keys/TESTkeys{}_{}.txt".format(keyNum, percentage/100),
                trainpospath,
                trainnegpath,
                percentage/100
            )
            print("Done!")

        print("Done making files!")


    def run_test(self, keyNum, pospath, negpath):

        for percentage in range(2,50,2):
        
            # counter variables
            total_checked = 0 
            correct = 0
            true_positives = 0
            false_positives = 0
            false_negatives = 0

            print("Training algorithm with {}% of training data...".format(percentage/1000))
            self.classifier.train("vectors/vectors_keys{}_{}.txt".format(keyNum, percentage/1000))
            if self.classifier.__str__() == 'id3':
                self.classifier.buildTree()
            elif self.classifier.__str__() == 'randfor':
                self.classifier.random_forest(6, 9, True, 0.1)
            print("Running classification tests with {}% of training data...".format(percentage/1000))

            i = 0
            # Checking negative revs...
            for filename in os.listdir(negpath):
                if i < (percentage/1000)*12500:
                    path = os.path.join(negpath, filename)
                    print(path)
                    res = self.classifier.classify(path)
                    # If review was classified as negative and IS negative...
                    if res == False:
                        correct += 1
                    else:
                        false_negatives += 1
                    total_checked += 1
                    i+=1
                else:
                    break
            
            i = 0
            # Checking positive revs...
            for filename in os.listdir(pospath):
                if i < (percentage/1000)*12500:
                    path = os.path.join(pospath, filename)
                    print(path)
                    res = self.classifier.classify(path)
                    # If review was classified as positive and IS positive...
                    if res == True:
                        correct += 1
                        true_positives += 1
                    else:
                        false_positives += 1
                    total_checked += 1
                    i+=1
                else:
                    break
            
            # calcualte statistics
            print("Saving to file...")
            with open("test/{}/TEST_out_keys{}.txt".format(self.classifier, keyNum),"a") as f:
                accuracy = (correct/total_checked)
                precision = (true_positives)/(true_positives+false_positives)
                recall = (true_positives)/(true_positives+false_negatives)
                f.write("{},{},{},{}\n".format(
                    percentage/1000,
                    round(accuracy, 4),
                    round(precision, 4),
                    round(recall, 4)
                ))
            print("Done saving to file!")

        for percentage in range(5,101,5):
        
            # counter variables
            total_checked = 0 
            correct = 0
            true_positives = 0
            false_positives = 0
            false_negatives = 0

            print("Training algorithm with {}% of training data...".format(percentage))
            self.classifier.train("vectors/vectors_keys{}_{}.txt".format(keyNum, percentage/100))
            if self.classifier.__str__() == 'id3':
                self.classifier.buildTree()
            elif self.classifier.__str__() == 'randfor':
                self.classifier.random_forest(6, 9, True, 0.1)
            print("Running classification tests with {}% of training data...".format(percentage/100))

            i = 0
            # Checking negative revs...
            for filename in os.listdir(negpath):
                if i < (percentage/100)*12500:
                    path = os.path.join(negpath, filename)
                    print(path)
                    res = self.classifier.classify(path)
                    # If review was classified as negative and IS negative...
                    if res == False:
                        correct += 1
                    else:
                        false_negatives += 1
                    total_checked += 1
                    i+=1
                else:
                    break
            
            i = 0
            # Checking positive revs...
            for filename in os.listdir(pospath):
                if i < (percentage/100)*12500:
                    path = os.path.join(pospath, filename)
                    print(path)
                    res = self.classifier.classify(path)
                    # If review was classified as positive and IS positive...
                    if res == True:
                        correct += 1
                        true_positives += 1
                    else:
                        false_positives += 1
                    total_checked += 1
                    i+=1
                else:
                    break
            
            # calcualte statistics
            print("Saving to file...")
            with open("test/{}/TEST_out_keys{}.txt".format(self.classifier, keyNum),"a") as f:
                accuracy = (correct/total_checked)
                precision = (true_positives)/(true_positives+false_positives)
                recall = (true_positives)/(true_positives+false_negatives)
                f.write("{},{},{},{}\n".format(
                    percentage/100,
                    round(accuracy, 4),
                    round(precision, 4),
                    round(recall, 4)
                ))
            print("Done saving to file!")




"""def test(classifyfunc, pospath, negpath, limiter):
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
    print("Accuracy:", (correct/counter)*100, "%")"""

#id3 = ID3()
#ts = Tester(id3)
#ts.buildTestVectorFiles(50,"aclImdb/train/pos", "aclImdb/train/neg")
#ts.run_test(50,"aclImdb/train/pos", "aclImdb/train/neg")

nbc = NaiveBayesClassifier()
ts = Tester(nbc)
#ts.buildTestVectorFiles(50,"aclImdb/train/pos", "aclImdb/train/neg")
ts.run_test(50,"aclImdb/test/pos", "aclImdb/test/neg")

#nbc.train("vectors/vectors_keys100_100.txt")
#test(nbc.classify, "aclImdb/test/pos", "aclImdb/test/neg",10)

#rf = Random_Forest()
#ts = Tester(rf)
#ts.buildTestVectorFiles(40,"aclImdb/train/pos", "aclImdb/train/neg")
#ts.run_test(10,"aclImdb/test/pos", "aclImdb/test/neg")

