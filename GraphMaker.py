import matplotlib.pyplot as plt

for i in range(10,51,10):

    perc = []
    acc = []
    precis = []
    recall = []

    perc2 = []
    acc2 = []
    precis2 = []
    recall2 = []

    with open("test/randfor/TEST_out_keys{}.txt".format(i), "r") as f:
        lines = f.readlines()
        data = [line.split(",") for line in lines]
        data = sorted(data, key=lambda x : float(x[0]))
            
        perc = [float(line[0]) for line in data]
        acc = [1 - float(line[1]) for line in data]
        precis = [float(line[2]) for line in data]
        recall = [float(line[3]) for line in data]

    with open("test/randfor/TRAIN_out_keys{}.txt".format(i), "r") as f:
        lines = f.readlines()
        data = [line.split(",") for line in lines]
        data = sorted(data, key=lambda x : float(x[0]))
            
        perc2 = [float(line[0]) for line in data]
        acc2 = [1 - float(line[1]) for line in data]
        precis2 = [float(line[2]) for line in data]
        recall2 = [float(line[3]) for line in data]

    plt.plot(perc, acc, '-g', label="test data")
    plt.plot(perc2, acc2, '-b', label="train data")
    #plt.plot(perc, [min(acc) for _ in perc], '--r', label="expected error")

    plt.title("Random Forest Learning Curves - {} keys, 9 trees (ID3 Instances)".format(i))
    plt.xlabel("percentage of training data")
    plt.ylabel("error")
    plt.legend() # required to show the labels
    plt.show()