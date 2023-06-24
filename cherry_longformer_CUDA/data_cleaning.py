import os.path
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import codecs
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import word_tokenize
stop = stopwords.words('english')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def preprocess(df):
    print("Preprocessing data ...")
    df['fact'] = df['fact'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if not word in (stop)]))
    df['context'] = df['context'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if not word in (stop)]))
    return df

# splitting data file into train, val, test
def preprocess_and_split_data(full_ds_file,validation, cv_k,classification_type,shuffle,test_size):
    data_splits=[]
    full_dataset_df = pd.read_csv(full_ds_file, delimiter='\t')
    full_dataset_df = preprocess(full_dataset_df)

    print("Generating data splits ...")
    if validation == 'cv':  # split for Cross Validation
        testing_set_size = full_dataset_df.shape[0]//cv_k  #number of rows in the df over the number of folds in CV
        start_slicing_index =0
        for i in range(cv_k):
            end_slicing_index= start_slicing_index+testing_set_size
            test_df = full_dataset_df.iloc[start_slicing_index:end_slicing_index,:]  # splitting the first n rows into the testing data frame
            train_df = full_dataset_df.drop(full_dataset_df.index[start_slicing_index:end_slicing_index]) # removing the chosen testing set from the dataset and saving the remaining in the train
            start_slicing_index=end_slicing_index
            # resetting pandas df index (to avoid indexing problem when usinf .loc in the CustomDataset class
            test_df = test_df.reset_index(drop=True)
            train_df = train_df.reset_index(drop=True)
            ds = (train_df,test_df,test_df)
            data_splits.append(ds)


    else:  # Split for Holdout Validation
        train_df, test_df = train_test_split(full_dataset_df, test_size=test_size, shuffle=shuffle) # splitting full ds into 20% testing, and 80% training
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        ds = (train_df, test_df)
        data_splits.append(ds)

    # write the full cleaned ds just in case
    preprocessed_file_path = "data/classification_" + str(classification_type) + "/preprocessed_full_ds.csv"
    full_dataset_df.to_csv(preprocessed_file_path, sep='\t')
    print("Full cleaned and pre-processed dataset saved as: preprocessed_full_ds.csv")
    return data_splits

def calculate_similarity(sentence1, sentence2):
    embedding_1= model.encode(sentence1, convert_to_tensor=True)
    embedding_2 = model.encode(sentence2, convert_to_tensor=True)
    similarity_tensor = util.pytorch_cos_sim(embedding_1, embedding_2)
    similarity=similarity_tensor.data[0][0].item()
    return similarity

# removes redundant examples based on cosine similarity between facts in every two examples.
def remove_redundant(full_ds_file,similarity_threshold,redund_free_file):
    with codecs.open(full_ds_file,'r',encoding='utf8') as f:
        f.readline()
        lines = f.readlines()
        for i in range(len(lines)) :
            #print("working on line "+str(i))
            if not lines[i].endswith("redundant"):
                lines[i]= lines[i].strip()
                event_id1,cluster_id1,fact1,context1,label1 = lines[i].split('\t')
                for j in range(i+1,len(lines)):
                    if not lines[j].endswith("redundant"):
                        lines[j] = lines[j].strip()
                        event_id2, cluster_id2, fact2, context2, label2 = lines[j].split('\t')
                        if event_id2==event_id1:
                            if calculate_similarity(fact1,fact2)>=similarity_threshold:
                                lines[j]=lines[j]+"\tredundant"

        with codecs.open(redund_free_file,'w',encoding='utf8') as out:
            out.write('event_id' + '\t' + 'factlets_cluster_id' + '\t' + 'fact' + '\t' + 'context' + '\t' + 'label')
            for line in lines:
                if not line.endswith("redundant"):
                    out.write('\n'+line)
            out.close()
        f.close()



# cleans and splits data
def clean_data(raw_ds_path,validation_type, cv_k,remove_redundants,classification_type,shuffle,test_size):
    if remove_redundants:
        print("Cleaning dataset from duplicates ...")
        clean_ds_path = "data/classification_"+str(classification_type)+"/clean_data.csv"
        if not os.path.exists(clean_ds_path):
            remove_redundant(raw_ds_path,0.90,clean_ds_path)
        data_splits = preprocess_and_split_data(clean_ds_path,validation_type, cv_k,classification_type,shuffle,test_size)

    else:
        data_splits = preprocess_and_split_data(raw_ds_path, validation_type, cv_k,classification_type,shuffle,test_size)

    return data_splits

