import os
import sys
import json
import time
import random
import logging
import argparse
import subprocess
import numpy as np
from tqdm.auto import tqdm
from typing import Optional
from typing import List, Any, Dict

from utils import mlperf_log_utils as mll
from utils import parsing_helpers as ph

import torch
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP


import transformers
from transformers import Trainer
from transformers import set_seed
from transformers import BertConfig
from transformers import get_scheduler
from transformers import AutoTokenizer
from transformers import BertForMaskedLM
from transformers import HfArgumentParser
from transformers import BertTokenizerFast
from transformers import TrainingArguments
from transformers.trainer_utils import is_main_process
from transformers import DataCollatorForLanguageModeling
#from modeling_bert_regression import BertForMaskedLMAndMoleculeScores


logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)


try:
    from apex.optimizers import FusedLAMB
    have_apex = True
except ImportError:
    from torch.optim import AdamW
    have_apex = False

try:
    from warmup_scheduler import GradualWarmupScheduler
    have_warmup_scheduler = True
except ImportError:
    pass

#dict helper for argparse
class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

device = torch.device('cuda:{}'.format(local_rank))
#dist.init_process_group('nccl', rank=world_rank, world_size=world_size)

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch WSI')
        parser.add_argument('--output_dir', type=str, help='output dir')
        parser.add_argument('--train_data_dir', type=str, help='train data')
        parser.add_argument('--tokenizer_path', type=str, help='path to trained tokenizer')

        parser.add_argument('--train_data_size', default=100, type=int, help='data size train')
        parser.add_argument('--val_data_size', default=100, type=int, help='data size val')        
        parser.add_argument('--train_local_bs', default=10, type=int, help='local batch size train')
        parser.add_argument('--val_local_bs', default=10, type=int, help='local batch size val')        
        parser.add_argument('--max_epochs', default=1, type=int, help='max epochs')
        parser.add_argument('--start_lr', default=1e-3, type=float, help='learning rate')
        parser.add_argument('--adam_eps', default=1e-8, type=float, help='learning rate')
        parser.add_argument('--weight_decay', default=1e-6, type=float, help='learning rate')
        parser.add_argument('--lr_warmup_steps', default=0, type=int, help='lr_warmup_steps')
        parser.add_argument('--lr_warmup_factor', default=1., type=float, help='lr_warmup_factor')
        parser.add_argument('--lr_schedule', action=StoreDictKeyPair)        
        parser.add_argument('--logging_frequency', default=100, type=int, help='Frequency with which the training progress is logged')
        parser.add_argument('--validation_frequency', default=100, type=int, help='Frequency with which the model is validated')
        
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args


args = Options().parse()
if(world_rank==0): print(args)

log_file = os.path.normpath(os.path.join(args.output_dir, "mlperf_log.log"))
logger = mll.mlperf_logger(log_file, "mlmol", "ORNL")
logger.log_start(key = "init_start", sync = True)        
logger.log_event(key = "cache_clear")
logger.log_event(key = "world_size", value = world_size)
logger.log_event(key = "global_batch_size", value = (args.train_local_bs * world_size))
logger.log_event(key = "opt_base_learning_rate", value = args.start_lr)

training_args = TrainingArguments(output_dir= args.output_dir,
                                  overwrite_output_dir=True,
                                  num_train_epochs=1,
                                  per_device_train_batch_size=args.train_local_bs,
                                  save_steps=10000,
                                  save_total_limit=5,
                                  prediction_loss_only=True,
                                  learning_rate=args.start_lr,
                                  weight_decay=args.weight_decay,
                                                            )
##########################
##### DATA ###############
##########################

max_rank = torch.distributed.get_world_size()-1
start_time = time.time()
print("Loading dataset...")
train_dataset = torch.load(args.train_data_dir)
end_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)


# log size of datasets
print(world_rank, len(train_dataset))
logger.log_event(key = "requested_train_samples", value = args.train_data_size)
logger.log_event(key = "requested_val_samples", value = args.val_data_size)
logger.log_event(key = "train_samples", value = len(train_dataset))
logger.log_event(key = "opt_learning_rate_warmup_steps", value = args.lr_warmup_steps)
logger.log_event(key = "opt_learning_rate_warmup_factor", value = args.lr_warmup_factor)
logger.log_event(key = "opt_epsilon", value = args.adam_eps)
logger.log_event(key = "data load time", value = end_time - start_time)

##########################
##### MODEL ##############
##########################

# Set seed before initializing model.
seed = random.randrange(100) #42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
logger.log_event(key = "seed", value = seed)

config = BertConfig(
    vocab_size=16000,
    hidden_size=516,
    num_attention_heads=12,
    num_hidden_layers=12
)
model = BertForMaskedLM(config=config)

# OPT (Better to always use LAMB)
if have_apex:
    optimizer = FusedLAMB(model.parameters(), lr = args.start_lr, eps = args.adam_eps, weight_decay = args.weight_decay)
else:
    # Not recommend to use this better start lr = 5e-5
    optimizer = AdamW(model.parameters(), lr = args.start_lr, eps = args.adam_eps, weight_decay = args.weight_decay)


# SCHEDULER
scheduler_after = ph.get_lr_schedule(args.start_lr, args.lr_schedule, optimizer, last_step = 0)

# LR warmup
if args.lr_warmup_steps > 0:
    if have_warmup_scheduler:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=args.lr_warmup_factor,
                                           total_epoch=args.lr_warmup_steps,
                                           after_scheduler=scheduler_after)

    # Throw an error if the package is not found
    else:
        raise Exception(f'Requested {pargs.lr_warmup_steps} LR warmup steps '
                        'but warmup scheduler not found. Install it from '
                        'https://github.com/ildoonet/pytorch-gradual-warmup-lr')
else:
    scheduler = scheduler_after

##########################
##### TRAINING ###########
##########################

model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=[local_rank])

trainer = Trainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=train_dataset,
                  optimizers=(optimizer,scheduler),
    )

trainer.train()
trainer.save_model(args.output_dir)

# run done
logger.log_end(key = "run_stop", sync = True, metadata = {'status' : 'success'})
