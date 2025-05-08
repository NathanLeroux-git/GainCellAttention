from torch.utils.data import Dataset, DataLoader, random_split, Sampler, DistributedSampler
import torch.distributed as dist
import os
import torch
import numpy as np
from datasets import load_dataset, load_from_disk
import time
import random

from transformers import AutoTokenizer, DataCollatorForLanguageModeling


class openwebtext():
    def __init__(self, config):
        # poor man's data loader
        data_dir = os.path.join(config.data_dir, config.dataset)
        self.train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        self.block_size = config.block_size
        self.batch_size = config.batch_size
        print("Batch size is set to:", config.batch_size)
        self.device_type = config.device_type
        self.device = config.device

    def get_batch(self, split, mask=None):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])       
        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y, None
    
class TextTorchDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        idx = int(idx)
        tokens = torch.zeros(0, dtype=torch.int64)
        # Continue adding tokens from the next texts until we reach max_length
        # while_loop_iters = 0
        while tokens.size(0) < self.max_length:
            next_tokens = self.tokenizer(self.texts[idx]['text'],
                                        return_tensors="pt",
                                        truncation=True,
                                        max_length=(self.max_length - tokens.size(0))
                                        )["input_ids"].squeeze(0)
            tokens = torch.cat((tokens, next_tokens), dim=0)
            idx = (idx + 1) % len(self.texts)  # Move to the next text (circular index)
            # while_loop_iters += 1
        
        # Truncate to max_length
        tokens = tokens[:self.max_length]  
        # # 
        # start = time.time()
        # tokens = self.tokenizer(self.texts[idx]['text'],
        #                                     return_tensors="pt",
        #                                     truncation=True,
        #                                     padding='max_length',
        #                                     max_length=self.max_length,
        #                                     )["input_ids"].squeeze(0) 
        # stop = time.time()
        # print(f"time was {stop-start:.2f}s")   
        # print(tokens.shape)
        
        # Create targets by shifting tokens to the right by one position
        # tokens, targets = tokens[:-1], tokens[1:]

        # targets = tokens.clone()
        # targets[:-1] = tokens[1:]
        # targets[-1] = self.tokenizer.eos_token_id  # Assign EOS token or padding to the last token
        
        return tokens#, while_loop_iters

# This one does not work
# class ShuffledSampler(Sampler): 
#     def __init__(self, dataset_size, seed=0):
#         self.indices = np.arange(dataset_size)
#         print('Indices build successfully')
#         np.random.default_rng(seed).shuffle(self.indices)  # Shuffle once

#     def __iter__(self):
#         return iter(self.indices)

#     def __len__(self):
#         return len(self.indices)

class ShuffledSampler(Sampler):
    def __init__(self, dataset_size, seed=None):
        self.dataset_size = dataset_size
        if seed is not None:
            random.seed(seed)
    
    def __iter__(self):
        # Infinite random sampling with replacement
        while True:
            # Yield a random index for each __getitem__ call.
            yield random.randint(0, self.dataset_size - 1)
    
    def __len__(self):
        return self.dataset_size

class openwebtext_raw():
    def __init__(self, config):
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        print(f"Loading openwebtext dataset...")
        ds_train = load_dataset("openwebtext", num_proc=96, split='train[:90%]')
        ds_test = load_dataset("openwebtext", num_proc=96, split='train[90%:]')
        print("Dataset loaded successfully")
        test_set = TextTorchDataset(ds_test, tokenizer, max_length=config.block_size)
        train_set = TextTorchDataset(ds_train, tokenizer, max_length=config.block_size)        
        self.train_dataloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True) 
        print("Dataloader build successfully")

        self.device_type = config.device_type
        self.device = config.device

    def get_batch(self, split, model_mask):
        dataloader = self.train_dataloader if split == 'train' else self.test_dataloader
        x = next(iter(dataloader))
        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x = x.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
        if model_mask is not None:
            model_mask = model_mask.to(x.device)
        return x, x, model_mask

class slimpajama():
    def __init__(self, config):
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        print(f"Loading SlimPajama-627B dataset...")
        # ds = load_dataset("cerebras/SlimPajama-627B", num_proc=96)
        ds = load_from_disk("../datasets/texts/slimpajama/data/")
        print("Dataset loaded successfully")
        train_data = ds['train']
        print(f"Train set size is {len(train_data)}")
        test_data = ds['test']
        print(f"Test set size is {len(test_data)}")
        train_set = TextTorchDataset(train_data, tokenizer, max_length=config.block_size)
        test_set = TextTorchDataset(test_data, tokenizer, max_length=config.block_size)
        print("Torch dataset build successfully")
        train_sampler, test_sampler = ShuffledSampler(len(train_set), seed=config.seed_offset), ShuffledSampler(len(test_set), seed=config.seed_offset)
        self.train_dataloader = DataLoader(train_set, batch_size=config.batch_size, sampler=train_sampler)
        self.test_dataloader = DataLoader(test_set, batch_size=config.batch_size, sampler=test_sampler)
        self.device_type = config.device_type
        self.device = config.device

    def get_batch(self, split, model_mask):
        dataloader = self.train_dataloader if split == 'train' else self.test_dataloader
        x = next(iter(dataloader))
        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x = x.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
        if model_mask is not None:
            model_mask = model_mask.to(x.device)
        return x, x, model_mask

    
class _slimpajama_scatter():
    def __init__(self, config):
        assert config.batch_size % config.ddp_world_size== 0
        if config.master_process:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
            print(f"Loading SlimPajama-627B dataset...")
            # ds = load_dataset("cerebras/SlimPajama-627B", num_proc=96)
            ds = load_from_disk("../datasets/texts/slimpajama/data/")
            print("Dataset loaded successfully")
            train_data = ds['train']
            print(f"Train set size is {len(train_data)}")
            test_data = ds['test']
            print(f"Test set size is {len(test_data)}")
            train_set = TextTorchDataset(train_data, tokenizer, max_length=config.block_size)
            test_set = TextTorchDataset(test_data, tokenizer, max_length=config.block_size)
            print("Torch dataset build successfully")
            train_sampler, test_sampler = ShuffledSampler(len(train_set), seed=config.seed_offset), ShuffledSampler(len(test_set), seed=config.seed_offset)
            self.train_dataloader = DataLoader(train_set, batch_size=config.batch_size*config.ddp_world_size, sampler=train_sampler)
            self.test_dataloader = DataLoader(test_set, batch_size=config.batch_size*config.ddp_world_size, sampler=test_sampler)
        dist.barrier()
        self.batch_size = config.batch_size
        self.block_size = config.block_size
        self.master_process = config.master_process
        self.ddp_local_rank = config.ddp_local_rank
        self.ddp_world_size = config.ddp_world_size
        self.device_type = config.device_type
        self.device = config.device

    def get_batch(self, split, model_mask):
        if self.ddp_world_size==1:
            dataloader = self.train_dataloader if split == 'train' else self.test_dataloader
            x = next(iter(dataloader))
            # while_loop_iters = while_loop_iters.to(torch.float32).mean().item()
            if self.device_type == 'cuda':
                # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                x = x.pin_memory().to(self.device, non_blocking=True)
            else:
                x = x.to(self.device)
            if model_mask is not None:
                model_mask = model_mask.to(x.device)
            ### We don't need to shift the labels here because the loss is shifting under the hood, see class ForCausalLMLoss https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py
        else:
            x = torch.empty((self.batch_size, self.block_size), dtype=torch.int64, device=self.device)
            if self.master_process:
                dataloader = self.train_dataloader if split == 'train' else self.test_dataloader
                x_chunks = next(iter(dataloader))
                x_chunks = x_chunks.pin_memory()
                x_chunks = list(torch.chunk(x_chunks, self.ddp_world_size, dim=0))
                # Move each chunk to the correct GPU
                x_chunks = [chunk.to(self.device, non_blocking=True) for chunk in x_chunks]
            dist.barrier()
            # Now, scatter from master (src=0) to all processes.
            # On rank 0, scatter_list is the list of chunks; on other ranks, pass None.
            dist.scatter(tensor=x, scatter_list=x_chunks if self.master_process else None, src=0)
            if self.master_process:
                del x_chunks
            x = x.to(self.device, non_blocking=self.master_process)
            if model_mask is not None:
                model_mask = model_mask.to(x.device)

        return x, x, model_mask
    
class ambers_dataset():
    def __init__(self, config):    
        self.max_length = config.block_size        
        self.device = config.device
        self.device_type = config.device_type
        self.block_size = config.block_size
        dataset_path = "../datasets/texts/AmberDatasets/"
        # Download dataset from HF and save it locally, only if dataset_path does not already exist.
        self.save_AmberDatasets_to_disk(dataset_path) if not os.path.isdir(dataset_path) else None
    
        # Load from disk
        self.test_tokenized_dataset = load_from_disk(os.path.join(dataset_path, 'test/'))
        self.test_loader = DataLoader(self.test_tokenized_dataset,
                        batch_size=config.batch_size,
                        collate_fn=self.collate_fn,                                   
                    )
        print('Test set loaded successfully')
        
        # Load from disk
        self.train_tokenized_dataset = load_from_disk(os.path.join(dataset_path, 'train/'))
        self.train_loader = DataLoader(self.train_tokenized_dataset,
                        batch_size=config.batch_size,
                        collate_fn=self.collate_fn,                                  
                    )
        print('Train set loaded successfully')
    
    def collate_fn(self, data):
        return torch.cat([torch.tensor(x['token_ids']).unsqueeze(0) for x in data], dim=0)

    def get_batch(self, split, model_mask):   
        dataloader = self.train_loader if split=="train" else self.test_loader
        
        x = next(iter(dataloader))
        x = x[:, :self.block_size]
        y = x
        attention_mask = torch.ones_like(x)
        ### We don't need to shift the labels here because the loss is shifting under the hood, see class ForCausalLMLoss https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py
        
        attention_mask[:, -1].fill_(0)
            
        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y, attention_mask = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True), attention_mask.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y, attention_mask = x.to(self.device), y.to(self.device), attention_mask.to(self.device)  
            
        T = x.shape[-1]
        if model_mask is not None:
            model_mask = model_mask[:T, :T]
            attention_mask = (model_mask.unsqueeze(0) * attention_mask.unsqueeze(1)).unsqueeze(1).to(torch.float32)
                  
        return x, y, attention_mask
    
    def save_AmberDatasets_to_disk(self, path="../datasets/texts/AmberDatasets"):        
        # Create dir and intermediate directories
        os.makedirs(path)
                
        # Load dataset from huggingface or from cache.
        # dataset = load_dataset("LLM360/AmberDatasets", num_proc=96)
        train_dataset = load_dataset('LLM360/AmberDatasets', split='train[:99%]', num_proc=96)
        test_dataset = load_dataset('LLM360/AmberDatasets', split='train[99%:]', num_proc=96)
        # Save to disk
        test_dataset.save_to_disk(os.path.join(path, "test/"), num_proc=96)
        
        # Save to disk
        train_dataset.save_to_disk(os.path.join(path, "train/"), num_proc=96)
        pass
    
    
class smallm_corpus():
    def __init__(self, config):
        # super().__init__()       
        self.max_length = config.block_size        
        self.device = config.device
        self.device_type = config.device_type
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenized_pad_token = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        
        # Data collator is a function that can agregate sequences of different tokens length in batches. Includes padding.
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                return_tensors="pt",
                                                mlm=False,
                                                )
        
        tokenized = True
        dataset_path = "../datasets/texts/tokenized-smallm-corpus/cosmopedia/" if tokenized else "../datasets/texts/smallm-corpus/"
        # Download dataset from HF and save it locally, only if dataset_path does not already exist.
        self.save_smallm_to_disk(dataset_path, tokenize=tokenized) if not os.path.isdir(dataset_path) else None
        # Load from disk
        self.test_tokenized_dataset = load_from_disk(os.path.join(dataset_path, 'test/'))
        self.test_loader = DataLoader(self.test_tokenized_dataset,
                        batch_size=config.batch_size,
                        collate_fn=data_collator,                                  
                    )
        print('Test set loaded successfully')
        
        # Load from disk
        self.train_tokenized_dataset = load_from_disk(os.path.join(dataset_path, 'train/'))        
        self.train_loader = DataLoader(self.train_tokenized_dataset,
                        batch_size=config.batch_size,
                        collate_fn=data_collator,                                  
                    )
        print('Train set loaded successfully')
        
    def save_smallm_to_disk(self, path, tokenize=True):        
        # Create dir and intermediate directories
        os.makedirs(path)
                
        # Load dataset from huggingface or from cache.
        dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", num_proc=96).train_test_split(test_size=0.001)# , features='text' 
        
        # Pre-process and tokenize test split
        test_dataset = dataset['test'].remove_columns(['prompt', 'audience', 'format', 'seed_data'])

        if tokenize:
            test_dataset = test_dataset.map(self.tokenize_function,
                                                                batched=True,
                                                                load_from_cache_file=True,
                                                                num_proc=96, # num_proc to be adapted depending on system
                                                                ).remove_columns(['text'])
        # Save to disk
        test_dataset.save_to_disk(os.path.join(path, "test/"), num_proc=96)
        
        # Pre-process and tokenize train split
        train_dataset = dataset['train'].remove_columns(['prompt', 'audience', 'format', 'seed_data'])
        if tokenize:
            train_dataset = train_dataset.map(self.tokenize_function,
                                                                batched=True,
                                                                load_from_cache_file=True,
                                                                num_proc=96, # num_proc to be adapted depending on system
                                                                ).remove_columns(['text'])
        # Save to disk
        train_dataset.save_to_disk(os.path.join(path, "train/"), num_proc=96)
    
    def tokenize_function(self, text):
        return self.tokenizer(text["text"],
                        truncation=True,
                        padding=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                        )

    def get_batch(self, split, model_mask):   
        dataloader = self.train_loader if split=="train" else self.test_loader
        
        data = next(iter(dataloader))
        x, y, attention_mask = data['input_ids'], data['labels'], data['attention_mask']
        # dataloader = self.train_loader if split == 'train' else self.test_loader
        ### We don't need to shift the labels here because the loss is shifting under the hood, see function ForCausalLMLoss https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py
        
        attention_mask[:, -1].fill_(0)
            
        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y, attention_mask = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True), attention_mask.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y, attention_mask = x.to(self.device), y.to(self.device), attention_mask.to(self.device)  
            
        T = x.shape[-1]
        if model_mask is not None:
            model_mask = model_mask[:T, :T]
            attention_mask = (model_mask.unsqueeze(0) * attention_mask.unsqueeze(1)).unsqueeze(1).to(torch.float32)
                  
        return x, y, attention_mask    