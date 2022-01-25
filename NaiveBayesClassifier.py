class NaiveBayesClassifier:
    """Naive Bayes Classifier class"""

    def __init__(self):
        self.keys = []
        self.vectors = []
        self.hc = 0

    def train(self, trainingVectorsPath):
        """Method that trains the algorithm from the training vector file provided.

        trainingVectorsPath -- (str) path to the training vector file.
        """
        # Flush data structures for training...
        self.keys = []
        self.vectors = []

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
        # Σε πόσα μηνύματα εκπαίδευσης της κατηγορίας c=1
        # εμφανίζεται η λέξη που αντιστοιχεί στη Xi ;
        
        # Counters for every keyword in positive reviews:
        attr_counter_pos = [0 for _ in range(len(self.keys))]
        # Counter for every keyword in negative reviews:
        attr_counter_neg = [0 for _ in range(len(self.keys))]

        # For every training vector
        for vector in self.vectors:
            # Check every key attribute
            for i in range(len(self.keys)):
                # Ff review vector is positive:
                if vector[-1] == 1:
                    # if the training vector for a specific keyword
                    # matches the to-be-classified review vector:
                    if vector[i] == rev_vector[i]:
                        # Increase the keyword appearances
                        # in positive reviews
                        attr_counter_pos[i] += 1
                else:
                    # if the training vector for a specific keyword
                    # matches the to-be-classified review vector:
                    if vector[i] == rev_vector[i]:
                        # Increase the keyword appearances
                        # in positive reviews
                        attr_counter_neg[i] += 1
                            

        # The training vector files created have always
        # half positive - half negative training vectors,
        # thus we start with the probability that a training
        # vector is positive:
        pc1x = (1/2)
        # For every keyword
        for counter in attr_counter_pos:
            # If there exists at least one match:
            if counter != 0:
                # Multiply the probability of the review
                # being positive by the probability of 
                # the keyword matching considering that the
                # review is positive.
                pc1x *= counter / (len(self.vectors ) // 2)
            else:
                # If there are no matches for a specific keyword,
                # we use the Laplace Approximation method:
                pc1x *= 1 / ((len(self.vectors) // 2) + 2)

        # The same for the negative outcome:
        pc0x = (1/2)
        for counter in attr_counter_neg:
            if counter != 0:
                pc0x *= (counter / (len(self.vectors) // 2))
            else:
                pc0x *= (1 / ((len(self.vectors) // 2) + 2))

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