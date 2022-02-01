import os
from random import randint

from ID3 import ID3
from  NaiveBayesClassifier import NaiveBayesClassifier
from RandForest import Random_Forest
from DatasetExplorer import DatasetExplorer

class Tester:
    """Tester class for algorithm statistics"""
    
    def __init__(self, classifier=None):
        self.classifier = classifier

    def setClassifier(self, classifier):
        self.classifier = classifier

    def buildTestVectorFiles(self, keyNum, trainpospath, trainnegpath):
        """Method that constructs training data frames and stores them in .txt files.
        
        keyNum -- (int) number of keys
        trainpospath -- (str) path to positive training data folder
        trainnegpath -- (str) path to negative training data folder
        """

        # In the [0.002, 0.048] interval the training data used to train
        # the classifier is crucial to represent its learning curves
        # thus more sampling is required:
        for percentage in range(2,50,2):
            print("Building training vector file with {} keys on {}% of training data...".format(keyNum, percentage/10))
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
        
        # In the [0.05, 1.0] interval, the error rate is expected to 
        # have already converged, thus less sampling is required.
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

            print("Training algorithm with {}% of training data...".format(percentage/10))
            self.classifier.train("vectors/vectors_keys{}_{}.txt".format(keyNum, percentage/1000))
            if self.classifier.__str__() == 'id3':
                self.classifier.buildTree()
            elif self.classifier.__str__() == 'randfor':
                self.classifier.random_forest(6, 9, True, 0.1)

            i = 0
            # Checking negative revs...
            print("Checking negative test cases...")
            for filename in os.listdir(negpath):
                if i < (percentage/1000)*12500:
                    path = os.path.join(negpath, filename)
                    #print(path)
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
            print("DONE checking negatives!")

            
            i = 0
            # Checking positive revs...
            print("Checking positive test cases...")
            for filename in os.listdir(pospath):
                if i < (percentage/1000)*12500:
                    path = os.path.join(pospath, filename)
                    #print(path)
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
            print("DONE checking positives!")
            
            # calcualte statistics
            print("Saving to file...")
            with open("test/{}/out_keys{}.txt".format(self.classifier, keyNum),"a", encoding='utf-8') as f:
                accuracy = (correct/total_checked)
                precision = (true_positives)/(true_positives+false_positives) if true_positives != 0 else 0
                recall = (true_positives)/(true_positives+false_negatives) if true_positives != 0 else 0
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
            print("Running classification tests...")

            i = 0
            # Checking negative revs...
            print("Checking negative test cases...")
            for filename in os.listdir(negpath):
                if i < (percentage/100)*12500:
                    path = os.path.join(negpath, filename)
                    #print(path)
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
            print("DONE checking negatives!")
            i = 0
            # Checking positive revs...
            print("Checking positive test cases...")
            for filename in os.listdir(pospath):
                if i < (percentage/100)*12500:
                    path = os.path.join(pospath, filename)
                    #print(path)
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
            print("DONE checking positives!")
            
            # calcualte statistics
            print("Saving to file...")
            with open("test/{}/out_keys{}.txt".format(self.classifier, keyNum),"a", encoding='utf-8') as f:
                accuracy = (correct/total_checked)
                precision = (true_positives)/(true_positives+false_positives) if true_positives != 0 else 0
                recall = (true_positives)/(true_positives+false_negatives) if true_positives != 0 else 0
                f.write("{},{},{},{}\n".format(
                    percentage/100,
                    round(accuracy, 4),
                    round(precision, 4),
                    round(recall, 4)
                ))
            print("Done saving to file!")