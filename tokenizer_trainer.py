import time
from transformers import AutoTokenizer

def get_training_corpus(data):
    return (
        data[i : i + 1000]
        for i in range(0, len(data), 1000)
    )

def train_tokenizer(data_path, save_path)

    data_file = open(data_path, "r")
    lines = data_file.readlines()
    data = []
    for line in lines:
        line = line[:-1]
        data.append(line)
        
    training_corpus = get_training_corpus(data)
    initial_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    start_time = time.time()
    print("Tokenizer Training Start!")
    tokenizer = initial_tokenizer.train_new_from_iterator(training_corpus, 52000)
    end_time = time.time()
    print("Training Finished!")
    print("tokenizer training time is: " + str(end_time - start_time))
    
    tokenizer.save_pretrained(save_path+"/alloy-data-tokenizer")
    
    return tokenizer