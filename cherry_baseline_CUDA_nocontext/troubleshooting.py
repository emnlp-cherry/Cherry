
# TODO: change to adapt for three class classification tasks

import codecs
def check_if_dummy(predictions_file): # checks if the mode is dummy (predicting all positive or all negative)
    positive = 0
    negative = 0
    with codecs.open(predictions_file,'r',encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if float(line)>0.5:
                positive+=1
            else:
                negative+=1
        print("total number of samples = "+str(len(lines)))
        print("number of positives = "+str(positive))
        print("number of negatives = "+str(negative))

check_if_dummy("results/output.txt")
