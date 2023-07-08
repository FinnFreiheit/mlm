import torch
from datasets import load_dataset
from const import DEBUG, ps
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig  # hint for steps 2 and 5
from transformers import DataCollatorForLanguageModeling  # hint for step 4
from transformers import TrainingArguments, Trainer

SEED = 42
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
MASKING_PROBABILITY = 0.1
MAX_SEQ_LENGTH = 256



def loadData():
    """ Load train and test splits from ag_news. Randomly selected 10% of the training set as validation."""
    dataset = load_dataset("ag_news")
    dataset = dataset.shuffle(SEED)

    if DEBUG: dataset["train"] = dataset["train"].select(range(4000))

    ds = dataset["train"].train_test_split(test_size=0.1)
    dataset["train"] = ds["train"]
    dataset["val"] = ds["test"]

    if (DEBUG): print(ps, "Dataset:", dataset)

    return dataset


def preprocess_function(sample: Dict[str, Any], seq_len):
    """ text pre-processing."""
    return tokenizer(sample["text"], padding="max_length", truncation=True, max_length=seq_len)


if __name__ == "__main__":
    dataset = loadData()
    encoded_ds = dataset.map(
        preprocess_function, remove_columns=["label"], batched=True, fn_kwargs={"seq_len": MAX_SEQ_LENGTH}
    )
    if DEBUG: print(ps, "encoded Dataset:", encoded_ds)

    # ToDo Do you need to replace the padding token pad with the end of the sequence eos token? Why or why not?
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=MASKING_PROBABILITY)

    if DEBUG: print(ps, "Data Collator:", data_collator)
