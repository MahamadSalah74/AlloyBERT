from model_trainer import train_model

tokenized_data = "AlloysBERT/data/tokenized_dataset"
tokenizer = "AlloysBERT/alloy-data-tokenizer"
save_path = "AlloysBERT/model"
train_model(tokenized_data, tokenizer, save_path)