import time
import json
from transformers import LineByLineTextDataset

def get_json_data(data_path):
    data = []
    for line in open(data_path, 'r'):
        data.append(json.loads(line))
    
def get_text(paper_data):
    start_index = paper_data.find('title')
    end_index = paper_data.find('\')', start_index)
    start_index = paper_data.find(' ', start_index)
    title = paper_data[start_index+2: end_index-1]
    
    start_index = paper_data.find('description')
    end_index = paper_data.find('\')', start_index)
    start_index = paper_data.find(' ', start_index)
    abstract = paper_data[start_index+2: end_index-1]
    
    start_index = paper_data.find('authkeywords')
    end_index = paper_data.find('\')', start_index)
    start_index = paper_data.find(' ', start_index)
    key_words = paper_data[start_index+2: end_index-1]
    
    
    return title, abstract, key_words

def preprocess_data(data):
    
    titles = []
    abstracts = []
    key_words_list = []
    for paper in data:
        paper_text = paper['entry']
        title, abstract, key_words = get_text(paper_text)
        titles.append(title)
        abstracts.append(abstract)
        key_words_list.append(key_words)
        
    return titles, abstracts, key_words_list

def save_data(path, titles, abstracts, key_words):

    with open(path+'/abstracts.txt', 'w') as fp:
        fp.write('\n'.join(abstracts))
    
    with open(path+'/titles.txt', 'w') as fp:
        fp.write('\n'.join(abstracts))
        
    with open(path+'/keyword.txt', 'w') as fp:
        fp.write('\n'.join(abstracts))
        
def create_dataset(data_path, save_path):
    data = get_json_data(data_path)
    titles, abstracts, key_words = preprocess_data(data)
    save_data(save_path, titles, abstracts, key_words)
    return titles, abstracts, key_words

def create_and_save_tokenized_data(data_path, save_path):

    start_time = time.time()
    print("Creating tokenized data...")
    dataset = LineByLineTextDataset(
        tokenizer=full_tokenizer,
        file_path=data_path,
        block_size=128,
    )
    end_time = time.time()
    torch.save(dataset, save_path)
    print("creating and saving tokenized data took " + str(end_time - start_time) + " seconds!")
    