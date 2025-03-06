"""
File: utils/data.py
Author: Jeffrey Rivero
Email: jeff@check-ai.com
Created: 02/20/2025
Last Modified: 02/24/2025
Description: Handles dataset loading, processing, and collation for training language models.
             Includes DataConfig class, DataProcessor, and CustomDataCollator for efficient
             data preparation and batching.
"""

import torch
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for dataset loading and processing"""

    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    max_seq_length: int = 1024
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    validation_split_percentage: Optional[int] = 5
    streaming: bool = False
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_test_samples: Optional[int] = None


def is_valid_text(example: Dict[str, Any]) -> bool:
    """Standalone function for filtering valid texts"""
    return bool(example.get("text") and isinstance(example["text"], str))


def create_tokenization_function(tokenizer: PreTrainedTokenizer, max_length: int):
    """Create a proper tokenization function"""

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # We'll pad in the collator
            return_attention_mask=True,
        )

    return tokenize_function


class DataProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.chunk_size = 1000  # Add chunk processing
        
    def preprocess_function(self, examples):
        try:
            texts = examples["text"]
            if isinstance(texts, str):
                texts = [texts]
                
            # Process in chunks to avoid memory issues
            all_tokenized = {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }
            
            for i in range(0, len(texts), self.chunk_size):
                chunk = texts[i:i + self.chunk_size]
                tokenized = self.tokenizer(
                    chunk,
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding=False,
                    return_attention_mask=True,
                    return_tensors=None  # Return lists for easier concatenation
                )
                
                all_tokenized["input_ids"].extend(tokenized["input_ids"])
                all_tokenized["attention_mask"].extend(tokenized["attention_mask"])
                all_tokenized["labels"].extend(tokenized["input_ids"])  # Use input_ids as labels
                
            return all_tokenized
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise


def load_and_prepare_datasets(
    tokenizer: PreTrainedTokenizer,
    data_config: DataConfig,
) -> DatasetDict:
    """Load and prepare datasets with picklable functions"""

    # Create data processor instance
    processor = DataProcessor(tokenizer, data_config.max_seq_length)

    # Load raw datasets
    if data_config.dataset_name:
        raw_datasets = load_dataset(
            data_config.dataset_name,
            data_config.dataset_config_name,
            num_proc=data_config.preprocessing_num_workers,
        )
    else:
        data_files = {}
        if data_config.train_file:
            data_files["train"] = data_config.train_file
        if data_config.validation_file:
            data_files["validation"] = data_config.validation_file

        extension = (
            data_config.train_file.split(".")[-1] if data_config.train_file else "text"
        )
        if extension == "txt":
            extension = "text"

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            num_proc=data_config.preprocessing_num_workers,
        )

    # Process datasets, this is a bottle neck we need to handle this better
    processed_datasets = DatasetDict()
    for split, dataset in raw_datasets.items():
        # Apply sample limits before processing to save computation
        if split == "train" and data_config.max_train_samples is not None:
            dataset = dataset.select(
                range(min(len(dataset), data_config.max_train_samples))
            )
            logger.info(
                f"Limiting train dataset to {data_config.max_train_samples} examples"
            )
        elif split == "validation" and data_config.max_eval_samples is not None:
            dataset = dataset.select(
                range(min(len(dataset), data_config.max_eval_samples))
            )
            logger.info(
                f"Limiting validation dataset to {data_config.max_eval_samples} examples"
            )
        elif split == "test" and data_config.max_test_samples is not None:
            dataset = dataset.select(
                range(min(len(dataset), data_config.max_test_samples))
            )
            logger.info(
                f"Limiting test dataset to {data_config.max_test_samples} examples"
            )

        # Filter invalid texts using standalone function
        filtered = dataset.filter(
            is_valid_text,
            num_proc=data_config.preprocessing_num_workers,
            desc=f"Filtering {split} split",
        )

        # Apply preprocessing using class method
        processed = filtered.map(
            processor.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=data_config.preprocessing_num_workers,
            desc=f"Processing {split} split",
            batch_size=100,
        )

        processed_datasets[split] = processed

        # Log sample processed example and size
        if len(processed) > 0:
            logger.info(f"\nSample processed example from {split}:")
            sample = processed[0]
            for key, value in sample.items():
                if isinstance(value, (list, torch.Tensor)):
                    logger.info(f"  {key}: shape={len(value)}")
                else:
                    logger.info(f"  {key}: {value}")
            logger.info(f"Processed {split} dataset size: {len(processed)}")

    return processed_datasets


class CustomDataCollator:
    """Picklable data collator class"""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
            logger.warning("No pad token found, using eos token instead")
            
    def __call__(self, features):
        try:
            # Validate input
            if not features:
                raise ValueError("Empty batch received")
                
            # Get max length in batch
            max_length = max(len(x["input_ids"]) for x in features)
            batch_size = len(features)
            
            # Pre-allocate tensors on CPU (NOT on device)
            batch = {
                "input_ids": torch.full(
                    (batch_size, max_length), 
                    self.pad_token_id, 
                    dtype=torch.long
                ),
                "attention_mask": torch.zeros(
                    (batch_size, max_length), 
                    dtype=torch.long
                ),
                "labels": torch.full(
                    (batch_size, max_length), 
                    -100, 
                    dtype=torch.long
                )
            }
            
            # Fill tensors efficiently
            for i, feature in enumerate(features):
                # Make sure we're working with lists, not tensors
                input_ids = feature["input_ids"]
                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.tolist()
                    
                length = len(input_ids)
                
                # Copy data without creating new tensors
                batch["input_ids"][i, :length] = torch.tensor(input_ids, dtype=torch.long)
                batch["attention_mask"][i, :length] = 1
                batch["labels"][i, :length] = torch.tensor(input_ids, dtype=torch.long)
                
            return batch
            
        except Exception as e:
            logger.error(f"Error in data collation: {str(e)}")
            raise