import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from typing import Dict, Any

SEED = 42
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
MASKING_PROBABILITY = 0.1
MAX_SEQ_LENGTH = 256

DEBUG = True
ps = "\n==========================================================\n"

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




if __name__ == '__main__':

    """ Set up Data """

    dataset = loadData()
    encoded_ds = dataset.map(
        preprocess_function, remove_columns=["label"], batched=True, fn_kwargs={"seq_len": MAX_SEQ_LENGTH}
    )
    if DEBUG: print(ps, "encoded Dataset:", encoded_ds)

    # ToDo Do you need to replace the padding token pad with the end of the sequence eos token? Why or why not?
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=MASKING_PROBABILITY)

    if DEBUG: print(ps, "Data Collator:", data_collator)

    """ Set up Model """

    dropout_probability = 0.15
    # model.roberta.encoder.layer[i].output.dropout = ?

    # Update the dropout prob-ability of the output layer in each of the 6 encoder layers (from 0.1) to 0.15.
    config = AutoConfig.from_pretrained("distilroberta-base")
    if DEBUG: print(ps, "config", config)
    # config.attention_probs_dropout_prob = 0.15
    # ToDo Ganz dünnes Eis, müssen wir mal schauen ob das so stimmt.
    config.hidden_dropout_prob = 0.15
    if DEBUG: print(ps, "attention_probs_dropout_prob: ", config.attention_probs_dropout_prob)

    model = AutoModelForMaskedLM.from_config(config)
    if DEBUG: print(ps, "Model:", model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps")
    model.to(device)

    """ Set up Trainer"""

    training_args = TrainingArguments(
        output_dir="mlm_model",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        load_best_model_at_end=True,
        weight_decay=0.01,
        # use_mps_device=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_ds["train"],
        eval_dataset=encoded_ds["val"],
        data_collator=data_collator,
    )

    trainer.train()

    # Inference
    text = "E-mail scam targets police chief Wiltshire Police warns about <mask> after its fraud squad chief was targeted."

