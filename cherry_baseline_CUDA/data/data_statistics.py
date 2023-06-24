import codecs


def print_statistics_binary(data_file):
    events = dict()
    tot_negative = 0
    tot_positive = 0
    with codecs.open(data_file,'r',encoding='utf8') as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            fields =  line.split("\t")
            if not fields[1] in events:
                events[fields[1]]=dict()
                events[fields[1]]["neg_examples"] = 0
                events[fields[1]]["pos_examples"] = 0
            label = fields[-1]
            if label == "1":
                events[fields[1]]["pos_examples"] += 1
                tot_positive+=1
            else:
                events[fields[1]]["neg_examples"] += 1
                tot_negative+=1
        f.close()
    print("Number of unique events in the data set is: "+str(len(events)))
    print("Total number of positive examples in the data set is: "+str(tot_positive))
    print("Total number of negative examples in the data set is: "+str(tot_negative))
    print(events)

def print_statistics_multi(data_file):
    events = dict()
    class_0 = 0
    class_1 = 0
    class_2 = 0
    with codecs.open(data_file,'r',encoding='utf8') as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            fields =  line.split("\t")
            if not fields[1] in events:
                events[fields[1]]=dict()
                events[fields[1]]["class_0"] = 0
                events[fields[1]]["class_1"] = 0
                events[fields[1]]["class_2"] = 0
            label = fields[-1]
            if label == "0":
                events[fields[1]]["class_0"] += 1
                class_0+=1
            elif label == "1":
                events[fields[1]]["class_1"] += 1
                class_1+=1
            else:
                events[fields[1]]["class_2"] += 1
                class_2 += 1
        f.close()
    print("Number of unique events in the data set is: "+str(len(events)))
    print("Total number of class_0 examples in the data set is: "+str(class_0))
    print("Total number of class_1 examples in the data set is: "+str(class_1))
    print("Total number of class_2 examples in the data set is: " + str(class_2))
    print(events)



#print_statistics_binary("classification_1/preprocessed_full_ds.csv")
#print_statistics_binary("classification_2/preprocessed_full_ds.csv")
#print_statistics_multi("classification_3/preprocessed_full_ds.csv")
print_statistics_multi("classification_4/preprocessed_full_ds.csv")