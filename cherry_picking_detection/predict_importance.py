# STEP #4 this script runs best trained cherry model on the first statement of each cluster (except cluster -1)
# and saves importance scores and includes them in the generated .json file.

from transformers import AutoModelForSequenceClassification
from data_loading import prepare_data
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from src.helper_scripts.helpers import read_json


#Converts a tensor of logits into an array of probabilities by applying the sigmoid function
def get_probs_from_logits(logits,classification_type): # the logits tensor has NxC size where N is the number of examples in the batch (batch size) and C is the number of classes
    if classification_type <=2:
        probs = torch.sigmoid(logits.unsqueeze(-1))
    else:
        probs = F.softmax(logits, dim=1)  # dim=1: applies softmax on each row of the tensor (each row adds up to 1)
    return probs.detach().cpu().numpy()

def predict_importance(inference_dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained("allenai/longformer-large-4096", num_labels=1) #loading the pretrained longformer instance
    model.load_state_dict(torch.load("allenai_longformer-large-4096_lr_2e-05_val_loss_0.4767_ep_2.pt"))  # loading the trained cherry model weights on Longformer
    model.to(device)
    model.eval()
    probs_all = []
    w = open("predictions.txt", 'w')
    with torch.no_grad():
        i=0
        for seq, attn_masks, global_attention_mask in tqdm(inference_dataloader): # for each batch
            print("Running inference on sample "+str(i)+" out of "+str(len(inference_dataloader)))
            seq, attn_masks, global_attention_mask = seq.to(device), attn_masks.to(device), global_attention_mask.to(device)
            outputs = model(seq, attn_masks, global_attention_mask)
            probs = get_probs_from_logits(outputs.logits.squeeze(-1), classification_type=1).squeeze(-1)
            probs_all += probs.tolist()
            i+=1
    w.writelines(str(prob) + '\n' for prob in probs_all)
    w.close()

def combine_predictions_and_event_ids(predicitons_file,full_inference_data_df):
    # add predictions to the dataframe
    probs_test = pd.read_csv(predicitons_file, header=None)  # prediction probabilities
    #probs_test.iloc[:, 0] = probs_test.iloc[:, 0].str.replace("[","")  # removing the "[" and "]" from the columns (because instead of one value at each row (like in binary classification) we are dealing with list of prob at each row
    #probs_test.iloc[:, 0] = probs_test.iloc[:, 0].str.replace("]", "")
    probs_test = probs_test.astype(float)
    prediction_threshold =0.5
    probs_test['predicted_label'] = probs_test.iloc[:, 0].gt(prediction_threshold).astype(int)
    # preds_test = (probs_test >= prediction_threshold).astype('uint8')  # predicted labels using the above fixed threshold
    preds_test = probs_test['predicted_label']
    full_inference_data_df['predictions'] = preds_test.tolist()
    full_inference_data_df.to_csv('event_ids_and_predictions.csv')

def combine_all_in_json(predictions_and_event_ids,events_clustered_json):
    predictions_df = pd.read_csv(predictions_and_event_ids)
    data = read_json(events_clustered_json)
    events  = data['events']
    for i,event in enumerate(events):
        for cluster_id, cluster in event["grouped_statements"].items():
            updated_cluster = dict()
            updated_cluster["statements"] = cluster
            matching_row = predictions_df.loc[(predictions_df['event_id'] == int(event['event_id'])) & (predictions_df['cluster_id'] == int(cluster_id))] # get the row that matches the cluster ID and the event ID
            predicted_label = matching_row["predictions"].tolist()
            updated_cluster["importance"] =  str(predicted_label[0]) # add the corresponding prediction label to the cluster from the dataframe
            if cluster_id=="-1":
                updated_cluster["importance"] = "mixed" ### TODO: include real predictions from clsuter -1
            data['events'][i]["grouped_statements"][cluster_id] = updated_cluster    # replace the old cluster with the new structure that carries the prediciton label

    # rewrite the json file with the predicitons included
    with open("bias_analysis_events_clustered_wpredictions.json", 'w', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, indent=4)
        out.close()




batch_size=1
pretrained_model= "allenai/longformer-large-4096"
max_seq_len =512
global_attention_loc=[0,1]
padding_strategy= "max_length"
inference_dataloader, full_inference_data_df = prepare_data("bias_analysis_inference_ready.csv",batch_size, pretrained_model,max_seq_len,global_attention_loc,padding_strategy)
predict_importance(inference_dataloader)
combine_predictions_and_event_ids("predictions.txt",full_inference_data_df)
combine_all_in_json("event_ids_and_predictions.csv","bias_analysis_events_clustered.json")