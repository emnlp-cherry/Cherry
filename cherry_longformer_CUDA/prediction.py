import torch
from tqdm import tqdm
import pandas as pd
from torchmetrics import Accuracy,F1Score,Precision, Recall
from sklearn.metrics import accuracy_score, balanced_accuracy_score,average_precision_score,recall_score,classification_report,precision_score
import codecs
import torch.nn.functional as F

#Converts a tensor of logits into an array of probabilities by applying the sigmoid function
def get_probs_from_logits(logits,classification_type): # the logits tensor has NxC size where N is the number of examples in the batch (batch size) and C is the number of classes
    if classification_type <=2:
        probs = torch.sigmoid(logits.unsqueeze(-1))
    else:
        probs = F.softmax(logits, dim=1)  # dim=1: applies softmax on each row of the tensor (each row adds up to 1)
    return probs.detach().cpu().numpy()


#predict the probabilities on a dataset with or without labels and print the result in a file
def test_prediction(net, device, test_dataloader, with_labels, result_file,classification_type):
    net.eval()
    w = open(result_file, 'w')
    probs_all = []

    with torch.no_grad():  # means do not calculate gradients at the end of the forward pass (becasue we are using the model for inference not training, so we do not need gradients to backpropagate)
        if with_labels:
            for seq, attn_masks, global_attention_mask, _ in tqdm(test_dataloader):
                seq, attn_masks, global_attention_mask = seq.to(device), attn_masks.to(device), global_attention_mask.to(device)
                outputs = net(seq, attn_masks, global_attention_mask)
                probs = get_probs_from_logits(outputs.logits,classification_type)
                if classification_type<=2:
                    probs = probs.squeeze(-1)
                probs_all += probs.tolist()
        else:
            for seq, attn_masks, global_attention_mask in tqdm(test_dataloader):
                seq, attn_masks, global_attention_mask = seq.to(device), attn_masks.to(device), global_attention_mask.to(device)
                outputs = net(seq, attn_masks, global_attention_mask)
                probs = get_probs_from_logits(outputs.logits.squeeze(-1),classification_type).squeeze(-1)
                probs_all += probs.tolist()

    w.writelines(str(prob)+'\n' for prob in probs_all)
    w.close()



def evaluate(test_df,prediction_threshold,path_to_output_file,exp_dir,classification_type):

    labels_test = test_df['label']  # true labels
    if classification_type<=2:
        probs_test = pd.read_csv(path_to_output_file, header=None)  # prediction probabilities
        probs_test.iloc[:, 0] = probs_test.iloc[:, 0].str.replace("[", "")  # removing the "[" and "]" from the columns (because instead of one value at each row (like in binary classification) we are dealing with list of prob at each row
        probs_test.iloc[:, 0] = probs_test.iloc[:, 0].str.replace("]", "")
        probs_test = probs_test.astype(float)
        probs_test['predicted_label'] = probs_test.iloc[:, 0].gt(prediction_threshold).astype(int)
        #preds_test = (probs_test >= prediction_threshold).astype('uint8')  # predicted labels using the above fixed threshold
        preds_test = probs_test['predicted_label']
    else:
        probs_test = pd.read_csv(path_to_output_file, header=None)
        probs_test.iloc[:, 0] = probs_test.iloc[:, 0].str.replace("[", "") # removing the "[" and "]" from the columns (because instead of one value at each row (like in binary classification) we are dealing with list of prob at each row
        probs_test.iloc[:, 2] = probs_test.iloc[:, 2].str.replace("]", "")
        probs_test = probs_test.astype(float)
        probs_test['predicted_label'] = probs_test.idxmax(axis=1) # returns the column name that contains the max value in each row and writes the column names in a new column named "predicted_label"
        preds_test = probs_test['predicted_label']

    y_true = test_df['label'].tolist()
    y_pred = preds_test.tolist()

    preds_test= torch.tensor(preds_test) # converting predictions and labels into torch Tensors  instead of pandas Series to pass them to torch metrics
    labels_test = torch.tensor(labels_test.values)

    with codecs.open(exp_dir + "results.txt", 'w', encoding='utf8') as out:
        print("sklearn's accuracy score = " + str(accuracy_score(y_true, y_pred, normalize=True)))
        out.write("\nsklearn's accuracy score = " + str(accuracy_score(y_true, y_pred, normalize=True)))
        print("sklearn's balanced accuracy score = " + str(balanced_accuracy_score(y_true, y_pred)))
        out.write("\nsklearn's balanced accuracy score = " + str(balanced_accuracy_score(y_true, y_pred)))
        print("sklearn's adjusted balanced accuracy score = " + str(balanced_accuracy_score(y_true, y_pred, adjusted=True)))
        out.write("\nsklearn's adjusted balanced accuracy score = " + str(balanced_accuracy_score(y_true, y_pred, adjusted=True)))
        if classification_type>2:
            print("sklearn's macro precision score = " + str(precision_score(y_true, y_pred, average='macro')))
            out.write("\nsklearn's macro precision score = " + str(precision_score(y_true, y_pred, average='macro')))
        else:
            print("sklearn's macro precision score = " + str(average_precision_score(y_true, y_pred, average='macro')))
            out.write("\nsklearn's macro precision score = " + str(average_precision_score(y_true, y_pred, average='macro')))
        print("sklearn's macro recall score = " + str(recall_score(y_true, y_pred, average='macro')))
        out.write("\nsklearn's macro recall score = " + str(recall_score(y_true, y_pred, average='macro')))
        print('\033[96m')
        out.write("\nClassification Report:----------------------------")
        if classification_type>2:
            print(classification_report(y_true, y_pred, labels=[0, 1, 2]))
            out.write(classification_report(y_true, y_pred, labels=[0, 1, 2]))
        else:
            print(classification_report(y_true, y_pred, labels=[0, 1]))
            out.write(classification_report(y_true, y_pred, labels=[0, 1]))

        print('\033[0m')
        num_classes = 2 if classification_type<=2 else 3
        accuracy = Accuracy(average='macro', num_classes=num_classes)
        print("Torch metrics' Accuracy = " + str(accuracy(preds_test, labels_test)))
        out.write("\nTorch metrics' Accuracy = " + str(accuracy(preds_test, labels_test)))
        f1_score = F1Score(average='macro', num_classes=num_classes)
        print("Torch metrics' F1 = " + str(f1_score(preds_test, labels_test)))
        out.write("\nTorch metrics' F1 = " + str(f1_score(preds_test, labels_test)))
        precision = Precision(average='macro', num_classes=num_classes)
        print("Torch metrics' Precision = " + str(precision(preds_test, labels_test)))
        out.write("\nTorch metrics' Precision = " + str(precision(preds_test, labels_test)))
        recall = Recall(average='macro', num_classes=num_classes)
        print("Torch metrics' Recall = " + str(recall(preds_test, labels_test)))
        out.write("\nTorch metrics' Recall = " + str(recall(preds_test, labels_test)))

    out.close()
