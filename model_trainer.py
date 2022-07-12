import time
import torch
torch.cuda.is_available()
from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer

def train_model(data_path, tokenizer_path, save_path):

    config = BertConfig(
        vocab_size=52_000,
        max_position_embeddings=512,
        num_attention_heads=12,
        num_hidden_layers=12)

    model = BertForMaskedLM(config=config)

    print("Model has " + str(model.num_parameters()) + " parameters!")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )


    training_args = TrainingArguments(
        output_dir= "./temp_model",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,

    )

    start_time = time.time()
    print("Loading dataset...")
    dataset = torch.load(data_path)
    end_time = time.time()
    print("Loading dataset took " + str(end_time - start_time) + " seconds!")
          
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    trainer.train()
    
    trainer.save_model(save_path)

