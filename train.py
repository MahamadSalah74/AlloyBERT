from transformers import HfArgumentParser
from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import BertTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments
import os
import webdataset as wd
import sys
import transformers
import json
import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)

from transformers.trainer_utils import is_main_process
from transformers import set_seed
import torch
from torch.utils.data import DataLoader
from typing import Optional
from modeling_bert_regression import BertForMaskedLMAndMoleculeScores
import subprocess

import torch.distributed as dist
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import get_scheduler
from tqdm.auto import tqdm
import argparse
import numpy as np
import random

import transformers
from torch.nn.utils.rnn import pad_sequence
from typing import List, Any, Dict
import time

# mlperf logger
import utils.mlperf_log_utils as mll
from utils import parsing_helpers as ph

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
dist.init_process_group('nccl', rank=world_rank, world_size=world_size)

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch WSI')
        parser.add_argument('--output_dir', type=str, help='output dir')
        parser.add_argument('--train_data_dir', type=str, help='train data')
        parser.add_argument('--val_data_dir', type=str, help='val data')        
        parser.add_argument('--bert_config', type=str, help='output dir')

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

##########################
##### DATA ###############
##########################

max_rank = torch.distributed.get_world_size()-1

train_dataset = wd.Dataset(args.train_data_dir + '/train_000000_19_{0..%d}.tar' % max_rank, length=args.train_data_size, shuffle=True).decode('torch').rename(input_ids='pth', molecule_scores='json').map_dict(molecule_scores=lambda x: [float(x['drug']), float(x['synth']), float(x['sol']), float(x['docking'])]).shuffle(args.train_data_size)
val_dataset = wd.Dataset(args.val_data_dir + '/part-val-002003_{0..%d}.tar' % max_rank, length=args.val_data_size).decode('torch').rename(input_ids='inputs.pth', molecule_scores='score.pth', attention_mask='mask.pth', labels='labels.pth')

with open(args.bert_config + '/config.json', 'r') as f:
    tokenizer_config = json.load(f)
tokenizer = BertTokenizerFast.from_pretrained(args.bert_config, **tokenizer_config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

class ValidationDataCollatorWithPadding(transformers.data.data_collator.DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = []
        if "labels" in features[0]:
            for f in features:
                labels.append(f.pop("labels"))
        batch = super().__call__(features)

        if len(labels) > 0:
            labels = pad_sequence(labels, batch_first=True, padding_value=-100)
            batch["labels"] = labels
        return batch


train_dataloader = DataLoader(train_dataset,
                              batch_size=args.train_local_bs,
                              sampler=None,
                              collate_fn=data_collator,
                              num_workers=0)

val_dataloader = DataLoader(val_dataset,
                            batch_size=args.val_local_bs,
                            sampler=None,
                            collate_fn=ValidationDataCollatorWithPadding(tokenizer),
                            num_workers=0)

# log size of datasets
print(world_rank, len(train_dataloader), len(val_dataloader))
logger.log_event(key = "requested_train_samples", value = args.train_data_size)
logger.log_event(key = "requested_val_samples", value = args.val_data_size)
logger.log_event(key = "train_samples", value = len(train_dataloader))
logger.log_event(key = "val_samples", value = len(val_dataloader))
logger.log_event(key = "opt_learning_rate_warmup_steps", value = args.lr_warmup_steps)
logger.log_event(key = "opt_learning_rate_warmup_factor", value = args.lr_warmup_factor)
logger.log_event(key = "opt_epsilon", value = args.adam_eps)

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
    vocab_size=32768,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12
)
model = BertForMaskedLMAndMoleculeScores(config=config)

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

#if pargs.max_validation_steps is not None:
#    logger.log_event(key = "invalid_submission")

step = 0
epoch = 0
current_lr = scheduler.get_last_lr()[0]
stop_training = False
model.train()

logger.log_end(key = "init_stop", sync = True)
logger.log_start(key = "run_start", sync = True)
start = time.time()

while True:

    #start epoch
    logger.log_start(key = "epoch_start", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync=True)

    # epoch loop
    train_data_iter = iter(train_dataloader)

    for _ in range(0, len(train_dataloader)):
        batch_train = next(train_data_iter)
        batch_train = {k: v.to(device) for k, v in batch_train.items()}

        # forward
        outputs = model(**batch_train)
        loss = outputs.loss
        acc = outputs.mask_accuracy
        drug_mse = outputs.drug_mse
        synth_mse = outputs.synth_mse
        sol_mse = outputs.sol_mse
        docking_mse = outputs.docking_mse
        # Backprop
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()

        # Scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
                
        # step counter
        step += 1

        
        # log training
        if (step % args.logging_frequency == 0):

            avg_loss = loss.detach() / 1
            dist.all_reduce(avg_loss)
            avg_loss = avg_loss.item() / dist.get_world_size()

            avg_acc = acc.detach() / 1
            dist.all_reduce(avg_acc)
            avg_acc = avg_acc.item() / dist.get_world_size()

            avg_drug = drug_mse.detach() / 1
            dist.all_reduce(avg_drug)
            avg_drug = avg_drug.item() / dist.get_world_size()

            avg_synth = synth_mse.detach() / 1
            dist.all_reduce(avg_synth)
            avg_synth = avg_synth.item() / dist.get_world_size()

            avg_sol = sol_mse.detach() / 1
            dist.all_reduce(avg_sol)
            avg_sol = avg_sol.item() / dist.get_world_size()

            avg_docking = docking_mse.detach() / 1
            dist.all_reduce(avg_docking)
            avg_docking = avg_docking.item() / dist.get_world_size()

            logger.log_event(key = "learning_rate", value = current_lr, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.log_event(key = "train_accuracy", value = avg_acc, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.log_event(key = "train_drug_mse", value = avg_drug, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.log_event(key = "train_synth_mse", value = avg_synth, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.log_event(key = "train_sol_mse", value = avg_sol, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.log_event(key = "train_docking_mse", value = avg_docking, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.log_event(key = "train_loss", value = avg_loss, metadata = {'epoch_num': epoch+1, 'step_num': step})
            
        # log val
        if (step % args.validation_frequency == 0):

            ep_vl_loss, ep_vl_acc, ep_vl_drug, ep_vl_synth, ep_vl_sol, ep_vl_docking = 0, 0, 0, 0, 0, 0
            logger.log_start(key = "eval_start", metadata = {'epoch_num': epoch+1})

            model.eval()
            val_data_iter = iter(val_dataloader)
            
            with torch.no_grad():
                for _ in range(0, len(val_dataloader)):
                    batch_val = next(val_data_iter)

                    batch_val = {k: v.to(device) for k, v in batch_val.items()}
                    outputs = model(**batch_val)

                    ep_vl_loss += outputs.loss
                    ep_vl_acc += outputs.mask_accuracy
                    ep_vl_drug += outputs.drug_mse
                    ep_vl_synth += outputs.synth_mse
                    ep_vl_sol += outputs.sol_mse
                    ep_vl_docking += outputs.docking_mse

            ep_vl_loss = ep_vl_loss/len(val_dataloader)
            avg_vl_loss = ep_vl_loss.detach() / 1
            dist.all_reduce(avg_vl_loss)
            avg_vl_loss = avg_vl_loss.item() / dist.get_world_size()

            ep_vl_acc = ep_vl_acc/len(val_dataloader)
            avg_vl_acc = ep_vl_acc.detach() / 1
            dist.all_reduce(avg_vl_acc)
            avg_vl_acc = avg_vl_acc.item() / dist.get_world_size()

            ep_vl_drug = ep_vl_drug/len(val_dataloader)
            avg_vl_drug = ep_vl_drug.detach() / 1
            dist.all_reduce(avg_vl_drug)
            avg_vl_drug = avg_vl_drug.item() / dist.get_world_size()

            ep_vl_synth = ep_vl_synth/len(val_dataloader)
            avg_vl_synth = ep_vl_synth.detach() / 1
            dist.all_reduce(avg_vl_synth)
            avg_vl_synth = avg_vl_synth.item() / dist.get_world_size()

            ep_vl_sol = ep_vl_sol/len(val_dataloader)
            avg_vl_sol = ep_vl_sol.detach() / 1
            dist.all_reduce(avg_vl_sol)
            avg_vl_sol = avg_vl_sol.item() / dist.get_world_size()

            ep_vl_docking = ep_vl_docking/len(val_dataloader)
            avg_vl_docking = ep_vl_docking.detach() / 1
            dist.all_reduce(avg_vl_docking)
            avg_vl_docking = avg_vl_docking.item() / dist.get_world_size()

            logger.log_event(key = "eval_accuracy", value = avg_vl_acc, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.log_event(key = "eval_drug_mse", value = avg_vl_drug, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.log_event(key = "eval_synth_mse", value = avg_vl_synth, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.log_event(key = "eval_sol_mse", value = avg_vl_sol, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.log_event(key = "eval_docking_mse", value = avg_vl_docking, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.log_event(key = "eval_loss", value = avg_vl_loss, metadata = {'epoch_num': epoch+1, 'step_num': step})
            
            elapsed_time = (time.time() - start) /60.
            if(world_rank==0): print("EASY_VAL_LOG", step, int(elapsed_time), avg_vl_loss, avg_vl_acc, avg_vl_drug, avg_vl_synth, avg_vl_sol, avg_vl_docking)            
            
            #if (iou_avg_val >= pargs.target_iou):
            #    logger.log_event(key = "target_accuracy_reached", value = pargs.target_iou,
            #                     metadata = {'epoch_num': epoch+1, 'step_num': step})
            #    stop_training = True

            # set to train
            model.train()

            logger.log_end(key = "eval_stop", metadata = {'epoch_num': epoch+1})

        if stop_training:
            break

    # log the epoch
    logger.log_end(key = "epoch_stop", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync = True)
    epoch += 1

    if epoch >= args.max_epochs or stop_training:
        break

# run done
logger.log_end(key = "run_stop", sync = True, metadata = {'status' : 'success'})
