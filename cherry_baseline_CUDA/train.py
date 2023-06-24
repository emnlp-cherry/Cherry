import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import copy
from transformers import AdamW, get_linear_schedule_with_warmup,AutoModelForSequenceClassification
import torch.nn as nn
from prediction import test_prediction,evaluate
from visualization import visualize_loss


def test_model(trained_model_path,test_dataloader,pretrained_model,freeze_pretrained,dropout,hidden_size,device,prediction_threshold,test_df,exp_dir,classification_type):
    # load trained model
    path_to_output_file = exp_dir+'predictions.txt'
    if classification_type <=2:  # if binary classification
        net = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=1,hidden_dropout_prob = dropout)
    else:
        net = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=3,hidden_dropout_prob = dropout)

    print("\nLoading the weights of the model...")
    net.load_state_dict(torch.load(trained_model_path))
    net.to(device)

    # run trained model on testing data set and generate predictions
    print("\nPredicting on test data...")
    test_prediction(net=net, device=device, test_dataloader=test_dataloader, with_labels=True,result_file=path_to_output_file,classification_type=classification_type)  # set the with_labels parameter to False if your want to get predictions on a dataset without labels
    print("\nPredictions are available in : {}".format(path_to_output_file))

    # evaluate model's performance on the testing data set using classification metrics and generated predictions
    print("\nEvaluating best model's performance on testing dataset ...")
    evaluate(test_df, prediction_threshold,path_to_output_file,exp_dir,classification_type)


def calc_valid_loss(net, device, loss_func, dataloader,classification_type):
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            if classification_type > 2:
                labels = labels.type(torch.LongTensor)
            seq, attn_masks, token_type_ids, labels = seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            outputs = net(seq, attn_masks, token_type_ids)
            if classification_type <= 2:
                mean_loss += loss_func(outputs.logits.squeeze(-1), labels.float()).item()
            else:
                mean_loss += loss_func(outputs.logits.squeeze(-1), labels).item()
            count += 1

    return mean_loss / count

def train_for_n_epochs(net, loss_func, opti, lr,lr_scheduler,train_dataloader,val_dataloader, epochs, device, pretrained_model,exp_dir,classification_type):
    best_loss = np.Inf
    all_epochs = []
    train_losses = []
    val_losses = []

    #scaler = GradScaler()

    for ep in range(epochs):
        print("\nprint('\033[1;35;47m __________ Epoch #"+str(ep)+"__________ \033[0;0m\n")
        net.train()
        total_training_loss = 0.0  # summ of training loss over the batches

        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(train_dataloader)): #identical to for step, batch in enumerate(train_loder)
            # if we have more than two classes (multi-class), then we are using CrossEntropyLoss which requires to cast the labels to Long data type before sending them to the device (GPU)
            if classification_type>2:
                labels = labels.type(torch.LongTensor)
            # Sending training data to GPU (converting to CUDA tensors)
            seq, attn_masks, token_type_ids, labels = seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)

            # forward pass
            net.zero_grad()  # Clear gradients
            # Enables autocasting (i.e. mixed precision => trades little accuracy for memeory)
            with autocast():
                # Obtaining the logits from the model (sentence_pair_classification) these are the logits coming out of the classification layer
                outputs = net(seq, attn_masks, token_type_ids)
                # Computing loss (using either CrossEntropyLoss or BCELoss based on classification type)
                if classification_type<=2: # to make sure targets are in LongTensor data type before entered to the loss function in case we are using CrossEntropyLoss instead of BCELoss
                    loss = loss_func(outputs.logits.squeeze(-1), labels.float())
                else:
                    loss = loss_func(outputs.logits.squeeze(-1), labels)
                total_training_loss += loss.item()

            # backward pass
            # Backpropagating the gradients. Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            loss.backward() #scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            # Optimization step: scaler.step() first unscales the gradients of the optimizer's assigned params. If these gradients do not contain infs or NaNs, opti.step() is then called,otherwise, opti.step() is skipped.
            opti.step() #scaler.step(opti)
            #scaler.update() # Updates the scale for next iteration.
            lr_scheduler.step() # Adjust the learning rate based on the number of iterations.


        train_loss_epoch= total_training_loss/len(train_dataloader)    # compute avg training loss for each epoch
        print("\nAverage training loss in this epoch = " + str(train_loss_epoch))

        val_loss = calc_valid_loss(net, device, loss_func, val_dataloader,classification_type)  # Compute validation loss
        # tracking validation and training loss on every epoch for plotting/visualization
        train_losses.append(train_loss_epoch)
        val_losses.append(val_loss)
        all_epochs.append(ep)
        print("\nEpoch {} complete! Validation Loss : {}".format(ep + 1, val_loss))


        if val_loss < best_loss:
            print("\033[1;0;33m \nBest validation loss improved from {} to {}".format(best_loss, val_loss))
            print("\033[0;0m")
            net_copy = copy.deepcopy(net)  # save a copy of the model
            best_loss = val_loss
            best_ep = ep + 1

            # Saving the model
            modified_pretrained_model_name = pretrained_model.replace("/","_")  # to avoid [Errno 2] No such file or directory since "/" in the pretrained model name means a directory in file paths
            path_to_model = exp_dir + '{}_lr_{}_val_loss_{}_ep_{}.pt'.format(modified_pretrained_model_name, lr,round(best_loss, 5), best_ep)
            torch.save(net_copy.state_dict(), path_to_model)
            print("\nThe model has been saved in {}".format(path_to_model))

    del loss
    torch.cuda.empty_cache()
    visualize_loss(all_epochs,train_losses,val_losses,exp_dir)
    return path_to_model

def train_model(lr, dataloaders, epochs, device, pretrained_model,freeze_pretrained,dropout, hidden_size,prediction_threshold,test_dfs,exp_dir,classification_type):
    # net means neural net (model), and this is the connected layer used for classification (on top of BERT)
    if classification_type <=2:  # if binary classification
        net = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=1,hidden_dropout_prob = dropout)
        loss_func = nn.BCEWithLogitsLoss()  # define loss function: Binary Cross Entropy. With a sigmoid (you only need to give it the raw logits)
    else:
        net = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=3,hidden_dropout_prob = dropout)
        loss_func = nn.CrossEntropyLoss()  # loss is Cross Entropy (The input to Cross Entropy is the output of the Sigmoid or the softmax function (prob distribution), but in Pytorch's CrossEntropyLoss and BCEWithLogits loss, the softmax/sigmoid is already embedded in the loss function itself, so you just need to enter the raw logits as is without a softmax/sigmoid

    net.to(device) # send the model to the device
    opti = AdamW(net.parameters(), lr=lr, weight_decay=1e-2)  # define optimizr and pass paramters you want to optimize
    num_warmup_steps = 0   # define LR warmup steps

    for f,fold in enumerate(dataloaders):
        print("\nprint('\033[0;30;46m Fold #"+str(f)+"------------------------------------------------------------- \033[0;0m")
        # unpack fold's training an dtesting data
        (train_dataloader,  test_dataloader) = fold
        test_df = test_dfs[f]
        num_steps = len(train_dataloader)* epochs  # number of batches (i.e. length of the train_dataloder) * # of epochs
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps,num_training_steps=num_steps)

        # train and validate (i.e. fine tune) the pretrained model using this data
        trained_model_path = train_for_n_epochs(net, loss_func, opti, lr, lr_scheduler, train_dataloader, test_dataloader, epochs, device, pretrained_model,exp_dir,classification_type)

        # Test (predict) and evlauate
        test_model(trained_model_path,test_dataloader,pretrained_model,freeze_pretrained,dropout,hidden_size,device,prediction_threshold,test_df,exp_dir,classification_type)

    return trained_model_path