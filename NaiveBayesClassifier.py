class NaiveBayesClassifier:
    """Naive Bayes Classifier class"""

    def __init__(self):
        self.keys = []
        self.vectors = []
        self.key_appearances = []
        self.positives = 0


    def train(self, trainingVectorsPath):
        """Method that trains the algorithm from the training vector file provided.

        trainingVectorsPath -- (str) path to the training vector file.
        """
        # Flush data structures for training...
        self.keys = []
        self.vectors = []
        self.key_appearances = [[[0,0],[0,0]] for _ in self.keys]

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

        # Prepare the appearance counters for each keyword:
        self.key_appearances = [[[0,0],[0,0]] for _ in self.keys]

        # Calculate the positive training vector counter:
        self.positives = sum([1 for vec in self.vectors if vec[-1] == 1])

        for index in range(len(self.keys)):
            for vector in self.vectors:
                # If training vector is negative
                if vector[-1] == 0:
                    # If vector does not have key
                    if vector[index] == 0:
                        # ex. key_appearances[index('BAD')][0][0]
                        #     == number of training vectors that  
                        #        are NEGATIVE and DON'T have the
                        #        keyword 'BAD'

                        # ex. key_appearances[index('BAD')][0][1]
                        #     == number of training vectors that  
                        #        are NEGATIVE and DO have the
                        #        keyword 'BAD'

                        # ex. key_appearances[index('BAD')][1][0]
                        #     == number of training vectors that  
                        #        are POSITIVE and DON'T have the
                        #        keyword 'BAD'

                        # ex. key_appearances[index('BAD')][1][1]
                        #     == number of training vectors that  
                        #        are POSITIVE and DO have the
                        #        keyword 'BAD'
                        self.key_appearances[index][0][0] += 1
                    else:
                        self.key_appearances[index][0][1] += 1
                else:
                    # If vector does not have key
                    if vector[index] == 0:
                        self.key_appearances[index][1][0] += 1
                    else:
                        self.key_appearances[index][1][1] += 1
            
    
    def classify(self, revpath):
        """Main classification method that classifies a review as positive (True) or negative (False)
        
        revpath - (str) path to the review file"""
        
        # Initialize a dummy training vector full of 0s.     
        rev_vector = [0 for _ in range(len(self.keys)+1)]
        
        # Open the review file:
        with open(revpath, "r", encoding='utf-8') as revfile:
            rev_text = revfile.read()
            words = rev_text.split(" ")
            for word in words:
                cleanWord = word.strip(".,!").upper()
                if cleanWord in self.keys:
                    # Vector-ify the review text...
                    # ex. review_vector[indexOf('BAD')] = 1
                    rev_vector[self.keys.index(cleanWord)] = 1
        
        # if P(C = 1 | X) > P(C = 0 | X)
        #   RETURN 1

        #P(C=1|X):
        #P(X=xi | C=1):
        # In how many training examples that are positive (c=1)
        # does the keyword X appear (xi = 1) or not (xi = 0).
        # (according to the value xi of the to-be-classified vector).
        
    
        # Begin by calculating the probability of
        # a positive training vector among the training
        # vectors.
        pc1x = self.positives / len(self.vectors)
        # For every keyword
        for i in range(len(self.keys)):
            # Multiply the product by the probability P(X=xi | C=1):
            # P(X=xi | C=1) = #appearances in positive vectors with Xi = rev_vector[i] / #positive vectors
            # Lastlyt, Laplace approximation (in case the #appearances in positive vectors is 0)
            pc1x *= (1 + self.key_appearances[i][1][rev_vector[i]]) / (self.positives + 2)

        # The same for the negative outcome:
        pc0x = (len(self.vectors)-self.positives)/len(self.vectors)
        for i in range(len(self.keys)):
            pc0x *= (1 + self.key_appearances[i][0][rev_vector[i]]) / ((len(self.vectors) - self.positives) + 2)


        # If the probability of the to-be-classified review
        # being positive is greater than that of being negative,
        # classify as positive
        if pc1x > pc0x:
            return True
        else:
            # Otherwise as negative:
            return False

    def __str__(self):
        """toString method as an identifier"""
        return "nbc"