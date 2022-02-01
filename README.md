# imdb-review-nlp (beta)
Natural Language Processing &amp; Machine Learning Algorithms over IMDB's Review database.

Authors; [Georgios E. Syros](https://www.github.com/gsiros "Georgios' GitHub"), [Anastasios Toumazatos](https://www.github.com/toumazatos "Tasos' GitHub"), [Evgenios Gkritsis](https://www.github.com/eGkritsis "Evgenios' GitHub")


## Batch Classification

### 1) Prepare the algorithms.

Before training the algorithms, we need to create different
training files with different % of training vectors.

```
>>> from Tester import Tester
>>> ts = Tester()
>>> ts.buildTestVectorFiles(X,"aclImdb/train/pos", "aclImdb/train/neg")
```

The buildTestVectorFiles will create training files with X keyword features
from the training folders provided by the dataset. These files are stored 
in the "vectors/" folder and look like "vectors_keysX_Y.txt" (Y = percentage
of training data). 

### 2) Test the algorithms.

After the creation of the training files, we provide the algorithm that we
want to perform tests on with the 'setClassifier' method. 

Finally, can execute the "run_test" method
to test the algorithm of choice on train/test data:

```
>>> from NaiveBayesClassifier import NaiveBayesClassifier
>>> nbc = NaiveBayesClassifier()
>>> ts.setClassifier(nbc)
>>> ts.run_test(X,"aclImdb/test/pos", "aclImdb/test/neg")
>>> ts.run_test(X,"aclImdb/test/pos", "aclImdb/test/neg")
```

This method outputs result files in the "test/" folder under
each classifier folder (nbc/id3/randfor). The files look like
"res_out_keysX.txt".

These result files have entries as rows with the following columns;
percentage_of_train_data (from 0(0%) to 1(100%)), accuracy, precision, recall

## Single Classifications

 - NaiveBayesClassifier:

 Firstly, we need to train the algorithm with one of the training vector
 files that we have created with the Tester's 'buildTestVectorFiles' method.

 Finally, we can provide a review as a .txt file. ID3 then classifies the 
 review as positive or negative.
 ```
 >>> from NaiveBayesClassifier import NaiveBayesClassifier
 >>> nbc = NaiveBayesClassifier()
 >>> nbc.train("vectors/vectors_keysX_Y.txt")
 >>> nbc.classify("demo_review.txt")
 False
 ```

 - ID3:

 Firstly, we need to train the algorithm with one of the training vector
 files that we have created with the Tester's 'buildTestVectorFiles' method.
 Then we need build the decision tree by providing the default category for
 classification and the stop threshold.

 Finally, we can provide a review as a .txt file. ID3 then classifies the 
 review as positive or negative.
 ```
 >>> from ID3 import ID3
 >>> id3 = ID3()
 >>> id3.train("vectors/vectors_keysX_Y.txt")
 >>> id3.buildTree(preset=<True(Positive) or False(Negative)>, stop_threshold=<real number from 0 to 1>)
 >>> id3.classify("demo_review.txt")
 False
 ```

 - Random Forest:

 Firstly, we need to train the algorithm with one of the training vector
 files that we have created with the Tester's 'buildTestVectorFiles' method.

 Secondly, we need to execute the 'random_forest' method to create the trees
 of the forest by providing different types of parameters.

 Finally, we can provide a review as a .txt file. ID3 then classifies the 
 review as positive or negative.
 ```
 >>> from RandForest import Random_Forest
 >>> rf = Random_Forest()
 >>> rf.train("vectors/vectors_keysX_Y.txt")
 >>> rf.random_forest(<number of m random keywords>, <number of trees (forest size)>, preset=<True(Positive) or False(Negative)>, stop_threshold=<real number from 0 to 1>)
 >>> rf.classify("demo_review.txt")
 False
 ```
