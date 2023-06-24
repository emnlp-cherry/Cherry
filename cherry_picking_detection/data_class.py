from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch


class CustomDataset(Dataset):

    def __init__(self, data_frame, pretrained_model, maxlen, global_attention_location, padding_strategy):

        self.data = data_frame  # datatype should be a pandas dataframe
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model,truncation_side='right')  ## find more arguments you can use at : https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast
        self.maxlen = maxlen
        self.global_attention_location = global_attention_location
        self.padding_strategy = padding_strategy

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting fact and context at the specified index in the data frame
        fact = str(self.data.loc[index, 'fact'])
        context = str(self.data.loc[index, 'context'])
        if len(fact) > 500:  # Truncate very long facts to avoid --> Exception: Truncation error: Sequence to truncate too short to respect the provided max_length
            fact = fact[:500]  # This does not happen in BERT because its tokenizer handles this case while Roberta's tokenizer does not.
        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(fact, context,
                                      padding=self.padding_strategy,  # Pad to max_length
                                      truncation="only_second",
                                      # 'only_second', #True,  # Truncate to max_length and truncates the context only
                                      max_length=self.maxlen,
                                      return_tensors='pt')  # Return torch.Tensor objects

        token_ids = encoded_pair['input_ids'].squeeze(0)  # (tensor of token ids)  squeeze changes the tensor from 2d to 1d e.g: [[0,1,5,3]]--> [0,1,5,3]
        # test a sample of how BERT tokenizer works
        # text_after_tokization(fact,context,token_ids,self.tokenizer)  # comment this line when finish inspecting tokenization output
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        sequence_ids = encoded_pair.encodings[0].sequence_ids  # binary list with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens, None for special tokens, we will use them to create global attentions mask only. we will not pass them to the model as in BERT
        # get a tensor with global attention locations (0: local/sliding window attention, 1: global attention)
        global_attention_mask = self.set_global_attention(sequence_ids)
        # RoBERTa tokenizer and Longformer model do not deal with token_type_ids so it is not used here


        return token_ids, attn_masks, global_attention_mask

    def set_global_attention(self, sequence_ids):
        # creating a list of zeros with the same length as token_type_ids
        global_attention_mask = [0] * len(sequence_ids)
        if 0 in self.global_attention_location:
            global_attention_mask[0] = 1
        if 1 in self.global_attention_location:
            for i in range(len(sequence_ids)):
                if sequence_ids[i] == 0:  # if the token belongs to the first sequence
                    global_attention_mask[i] = 1  # set the global attention to True for that token
        if 2 in self.global_attention_location:
            for i in range(len(sequence_ids)):
                if sequence_ids[i] == 1:
                    global_attention_mask[i] = 1

        # convert the global attention mask from a list to a FloatTensor (required by the model's forward method implementation)
        global_attention_mask = torch.FloatTensor(global_attention_mask)

        return global_attention_mask
