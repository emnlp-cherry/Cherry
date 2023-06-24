# methods that can be called temporarily to help debuggin and inspecting the different outputs of differnt stages
# of the ML pipeline

def text_after_tokization(fact, context, token_ids,tokenizer):
    print("Fact: "+fact)
    print("Context: "+context+'\n#########################################################################')
    tokenized_text = ""
    vocab = tokenizer.vocab
    for i,id in enumerate(token_ids):
        for token, token_id in vocab.items():
            if id == token_id:
                tokenized_text+=token+' '
                break
    print(tokenized_text)