import os
import random
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
from preprocess import preprocess_function, group_texts
from typing import Any, Optional
import pdb

def constants_like(X, const):
    constants = [[const for _ in row] for row in X]
    return constants

def create_dataset(
    data_path, 
    training_prop=0.8,
    validation_prop=0.1,
    testing_prop=0.1
    ):
    
    samples = torch.load(data_path)
    num_samples = len(samples)
    num_train = int(num_samples * training_prop)
    num_validation = int(num_samples * validation_prop)
    num_test = num_samples - num_train - num_validation
    assert(num_train + num_validation < num_samples)
    df = {
        "train": {"text": samples[:num_train]},
        "val": {"text": samples[num_train:num_train+num_validation]},
        "test": {"text": samples[num_train+num_validation:]},
    }
    dataset_dict = {data_split: Dataset.from_dict(df[data_split]) for data_split in ["train", "val", "test"]}
    dataset = DatasetDict(dataset_dict)
    return dataset

def create_tokenized_dataset(
    data_path, 
    training_prop=0.8,
    validation_prop=0.1,
    testing_prop=0.1
    ):
    
    samples = torch.load(data_path)
    num_samples = len(samples)
    num_train = int(num_samples * training_prop)
    num_validation = int(num_samples * validation_prop)
    num_test = num_samples - num_train - num_validation
    assert(num_train + num_validation < num_samples)

    df = {
        "train": {"input_ids": samples[:num_train]},
        "val": {"input_ids": samples[num_train:num_train+num_validation]},
        "test": {"input_ids": samples[num_train+num_validation:]},
    }
    for data_split in ["train", "val", "test"]:
        df[data_split]["token_type_ids"] = constants_like(df[data_split]["input_ids"], 0)
        df[data_split]["attention_mask"] = constants_like(df[data_split]["input_ids"], 1)
        
    dataset_dict = {data_split: Dataset.from_dict(df[data_split]) for data_split in ["train", "val", "test"]}
    dataset = DatasetDict(dataset_dict)
    return dataset

def preprocess_dataset(dataset, tokenizer, block_size=None):
    dataset = dataset.flatten()
    preprocessed_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )
    
    if block_size:
        preprocessed_dataset = preprocessed_dataset.map(
            lambda x: group_texts(x, block_size),
            batched=True,
            num_proc=4,
        )

    return preprocessed_dataset

def create_data_collator(tokenizer, masking_method="all", mlm_probability=0.15, target_tokens=None):
    if masking_method == "all":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)
    elif masking_method == "random_target_token":
        assert(target_tokens is not None)
        data_collator = RandomTargetTokenDataCollatorForLanguageModeling(tokenizer=tokenizer, target_tokens=target_tokens)
    else:
        raise NotImplementedError("This masking method ({}) is not implemented!".format(masking_method))
    return data_collator

class RandomTargetTokenDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, target_tokens, mlm=True, mlm_probability=0.15):
        super().__init__(tokenizer, mlm, mlm_probability)
        self.target_tokens = target_tokens
    
    def torch_mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[Any] = None):
        # Create mask array, initially set to False
        mask = torch.full(inputs.shape, False, dtype=torch.bool)
        
        for i in range(inputs.size(0)):
            # Find positions that are target letters
            target_token_positions = []
            for j in range(inputs.size(1)):
                token = self.tokenizer.convert_ids_to_tokens(inputs[i, j].item())
                if token in self.target_tokens:
                    target_token_positions.append(j)
            
            # Randomly choose one position to mask if any
            if len(target_token_positions) > 0:
                mask_pos = random.choice(target_token_positions)
                mask[i, mask_pos] = True

        # Mask the chosen tokens
        labels = torch.where(
            mask, 
            inputs, 
            torch.tensor(-100, dtype=torch.long)
        )
        inputs = torch.where(
            mask,
            torch.tensor(self.tokenizer.mask_token_id, dtype=torch.long),
            inputs,
        )
        
        return inputs, labels
    

if __name__ == "__main__":    
    import pdb
    from transformers import AutoTokenizer
    input_data = ["-a-b", "--ef"]
    # tokenizer = AutoTokenizer.from_pretrained("./huggingface/pretrained_models/bert-base-cased")
    # encoding = tokenizer(input_data, return_tensors="pt", padding=True, truncation=True)
    # # pdb.set_trace()
    # custom_data_collator = RandomLetterDataCollatorForLanguageModeling(tokenizer=tokenizer)
    # # Extract the `input_ids` tensor from the tokenization output

    # # Apply custom data collator
    # print([encoding["input_ids"][i] for i in range(encoding["input_ids"].size(0))])
    # batch = custom_data_collator([encoding["input_ids"][i] for i in range(encoding["input_ids"].size(0))])
    # pdb.set_trace()

