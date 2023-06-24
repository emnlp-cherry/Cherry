###################################################################################################
# D A T A    P R E P A R A T I O N
###################################################################################################


from torch.utils.data import DataLoader
from data_class import CustomDataset

def prepare_data(data_splits, batch_size, pretrained_model,max_seq_len):
    print("Preparing and loading data...")
    dataloaders = []
    test_dfs =[]
    for data_split in data_splits:
        # keep only certain columns (fact, context, label)
        df_train = data_split[0][['fact', 'context','label']]
        df_test = data_split[1][['fact', 'context','label']]

        train_set = CustomDataset(df_train, with_labels=True, pretrained_model=pretrained_model,maxlen=max_seq_len)
        test_set = CustomDataset(df_test, with_labels=True, pretrained_model=pretrained_model,maxlen=max_seq_len)

        train_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=0,shuffle=True) ## Dataloaders accept argument of type torch.Dataset, hence, we created our own CustomDataset class that inherits torch.Dataset
        test_dataloader = DataLoader(test_set, batch_size=batch_size, num_workers=0,shuffle=False) # set num_workers to 0 when debugging to see debug variables values. dataloaders take care of batching, to facilitate debugging use one worker

        split_dataloaders = (train_dataloader,test_dataloader)
        dataloaders.append(split_dataloaders)
        test_dfs.append(df_test)

    return dataloaders,test_dfs