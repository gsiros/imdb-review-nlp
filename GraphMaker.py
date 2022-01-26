import matplotlib.pyplot as plt

#for i in range(10,51,10):

perc = []
acc = []
precis = []
recall = []

perc2 = []
acc2 = []
precis2 = []
recall2 = []
for i in range(30,31,10):
    with open("test/randfor/TEST_out_keys{}.txt".format(i), "r", encoding='utf-8') as f:
        lines = f.readlines()
        data = [line.split(",") for line in lines]
        data = sorted(data, key=lambda x : float(x[0]))
            
        perc = [float(line[0])*100 for line in data]
        acc = [(float(line[1]))*100 for line in data]
        precis = [float(line[2])*100 for line in data]
        recall = [float(line[3])*100 for line in data]

    with open("test/randfor/TRAIN_out_keys{}.txt".format(i), "r", encoding='utf-8') as f:
        lines = f.readlines()
        data = [line.split(",") for line in lines]
        data = sorted(data, key=lambda x : float(x[0]))
            
        perc2 = [float(line[0])*100 for line in data]
        acc2 = [(float(line[1]))*100 for line in data]
        precis2 = [float(line[2])*100 for line in data]
        recall2 = [float(line[3])*100 for line in data]

    f1_test = [(2*item[0]*item[1])/(item[0]+item[1]) for item in list(zip(precis, recall))]
    f1_train = [(2*item[0]*item[1])/(item[0]+item[1]) for item in list(zip(precis2, recall2))]
    
    print("TEST ==============================")
    for entry in list(zip(perc, f1_test)):
        print(entry[0], round(float(entry[1]),2))
    print("TRAIN ==============================")
    for entry in list(zip(perc, f1_train)):
        print(entry[0], round(float(entry[1]),2))

    #minerr = min(acc)
    plt.plot(perc, acc, '-b', label="learning curve")
    #plt.plot(perc, recall, '-r', label="recall")
    #plt.plot(perc, f1, '-g', label="F1")
    #plt.plot(perc, [min(acc) for _ in perc], '--r', label="minimum error \nachieved ({}%)".format(round(minerr,2)))

    plt.title("Random Forest Classifier - {} keys, 9 trees".format(i))
    plt.xlabel("training data (% of total)")
    plt.ylabel("percentage (%)")
    plt.legend() # required to show the labels
    plt.show()