def topwords(wordpath, ratingpath, limiter):
    words = []
    ratings = []
    with open(wordpath, "r") as wordfile:
        lines = wordfile.readlines()
        words.extend([line.strip("\n").upper() for line in lines])

    with open(ratingpath, "r") as ratingfile:
        lines = ratingfile.readlines()
        ratings.extend([float(line.strip("\n")) for line in lines])

    unified = list(zip(words, ratings))
    unified = list(sorted(unified, key= lambda x: x[1]))
    i = 0
    for entry in unified:
        if abs(entry[1]) < 5:
            print(entry[0], entry[1])
            i += 1
        else:
            break
topwords('aclImdb/imdb.vocab', 'aclImdb/imdbEr.txt', 100)
