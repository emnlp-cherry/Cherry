from torch.utils.data import Dataset
from transformers import AutoTokenizer
from inspection import text_after_tokization
class CustomDataset(Dataset):

    def __init__(self,data_frame, with_labels, pretrained_model, maxlen):

        self.data = data_frame  # datatype should be a pandas dataframe
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, truncation_side='right')## find more arguments you can use at : https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast
        self.maxlen = maxlen
        self.with_labels = with_labels

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
                                      padding='max_length',  # Pad to max_length
                                      truncation='only_second', #True,  # Truncate to max_length and truncates the context only
                                      max_length=self.maxlen,
                                      return_tensors='pt')  # Return torch.Tensor objects

        token_ids = encoded_pair['input_ids'].squeeze(0)  # (tensor of token ids)  squeeze changes the tensor from 2d to 1d e.g: [[0,1,5,3]]--> [0,1,5,3]
        # test a sample of how BERT tokenizer works
        #text_after_tokization(fact,context,token_ids,self.tokenizer)  # comment this line when finish inspecting tokenization output
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.loc[index, 'label']
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids


    