from torch.utils.data import DataLoader
from data_class import CustomDataset
import pandas as pd
from src.helper_scripts.helpers import read_json
import codecs
import json
# STEP 3: add ids for events to combine predictions with events based on event ids later after inference.
# generate a csv with only necessary columns for predicitons in order to load data with pandas dataframes at infernece time
# prepare data and load data as CustomDataset objects into data loaders for inference.

def prepare_data( inference_data_file_csv, batch_size, pretrained_model,max_seq_len,global_attention_loc,padding_strategy):
    print("Preparing and loading data...")

    full_inference_data_df = pd.read_csv(inference_data_file_csv, delimiter='\t')
    # keep only certain columns (fact, context)
    inference_data_df = full_inference_data_df[['fact', 'context']]
    inference_set = CustomDataset(inference_data_df, pretrained_model=pretrained_model, maxlen=max_seq_len,
                              global_attention_location=global_attention_loc, padding_strategy=padding_strategy)
    inference_dataloader = DataLoader(inference_set, batch_size=batch_size, num_workers=0,shuffle=False)  # set num_workers to 0 when debugging to see debug variables values. dataloaders take care of batching, to facilitate debugging use one worker

    return inference_dataloader, full_inference_data_df

def add_event_id(bias_analysis_clustered_json):  # to rpeserve order
    data = read_json(bias_analysis_clustered_json)
    for i,e in enumerate(data['events']):
        data['events'][i]["event_id"] = str(i)# add event id to each event
    with open("bias_analysis_events_clustered.json", 'w', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, indent=4)
        out.close()


def convert_json_to_csv(bias_analysis_clustered_json):  # to be able to use pandas and dataloaders when inferencing using the model
    data = read_json(bias_analysis_clustered_json)
    events  = data['events']
    with codecs.open("bias_analysis_inference_ready.csv",'w',encoding='utf8') as out:
        out.write('event_id' + '\t' + 'cluster_id' + '\t' + 'fact' + '\t' + 'context' + '\n')
        for e in events:
            context = e["context"]
            context = context.replace('\t', '')
            context = context.replace('\n', ' ')
            context = context.replace('\r', ' ')
            context = context.replace("\"", "“")
            context = context.strip()
            for cluster_id, cluster in e["grouped_statements"].items():
                if not cluster_id=="-1":
                    statement = cluster[0]['text']
                    statement = statement.replace('\t', '')
                    statement = statement.replace('\n', ' ')
                    statement = statement.replace('\r', ' ')
                    statement = statement.replace("\"", "“")
                    statement = statement.strip()
                    out.write(e["event_id"]+'\t'+cluster_id+'\t'+statement+'\t'+context+'\n')
                elif cluster_id =="-1":   #remove this block if you don't want to inference on cluster -1
                    for s, statement in enumerate(cluster):
                        statement = cluster[0]['text']
                        statement = statement.replace('\t', '')
                        statement = statement.replace('\n', ' ')
                        statement = statement.replace('\r', ' ')
                        statement = statement.replace("\"", "“")
                        statement = statement.strip()
                        out.write(e["event_id"]+'\t'+cluster_id+'\t'+statement+'\t'+context+'\n')
                        if s==20:
                            break
        out.close()







#add_event_id("bias_analysis_events_clustered.json")
#convert_json_to_csv("bias_analysis_events_clustered.json")
