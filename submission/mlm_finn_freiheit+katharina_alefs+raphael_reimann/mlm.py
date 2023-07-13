import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import pipeline
from typing import Dict, Any
import sys
import wandb
import math

"""
BATCH_SIZE = 32
NUMOFEPOCH = 3
WEIGHTDECAY = 0.01
LEARNINGRATE = 2e-5"""

BATCH_SIZE = int(sys.argv[1])
NUMOFEPOCH = int(sys.argv[2])
WEIGHTDECAY = float(sys.argv[3])
LEARNINGRATE = float(sys.argv[4])

SEED = 42
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
MASKING_PROBABILITY = 0.1
MAX_SEQ_LENGTH = 256

DEBUG = True
ps = "\n==========================================================\n"


def printArgs():
    print("Training Arguments: \n")
    print("Batch size: ", BATCH_SIZE, "typ: ", type(BATCH_SIZE))
    print("Num of Epoch: ", NUMOFEPOCH, "typ: ", type(NUMOFEPOCH))
    print("Weight Decay: ", WEIGHTDECAY, "typ: ", type(WEIGHTDECAY))
    print("Lerning Rate: ", LEARNINGRATE, "typ: ", type(LEARNINGRATE))


def loadData():
    """ Load train and test splits from ag_news. Randomly selected 10% of the training set as validation."""
    dataset = load_dataset("ag_news")
    dataset = dataset.shuffle(SEED)

    #if DEBUG: dataset["train"] = dataset["train"].select(range(4000))

    ds = dataset["train"].train_test_split(test_size=0.1)
    dataset["train"] = ds["train"]
    dataset["val"] = ds["test"]

    if (DEBUG): print(ps, "Dataset:", dataset)

    return dataset


def preprocess_function(sample: Dict[str, Any], seq_len):
    """ text pre-processing."""
    return tokenizer(sample["text"], padding="max_length", truncation=True, max_length=seq_len)


if __name__ == '__main__':

    printArgs()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="mlm-first-test",
    )

    """ Set up Data """

    dataset = loadData()
    encoded_ds = dataset.map(
        preprocess_function, remove_columns=["label"], batched=True, fn_kwargs={"seq_len": MAX_SEQ_LENGTH}
    )
    if DEBUG: print(ps, "encoded Dataset:", encoded_ds)

    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=MASKING_PROBABILITY)

    if DEBUG: print(ps, "Init Data Collator")

    """ Set up Model """
    # Update the dropout prob-ability of the output layer in each of the 6 encoder layers (from 0.1) to 0.15.
    config = AutoConfig.from_pretrained("distilroberta-base")
    if DEBUG: print(ps, "config", config)

    model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")
    if DEBUG: print(ps, "Model:", model)

    for layer in model.roberta.encoder.layer:
        layer.output.dropout.p = 0.15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps")
    model.to(device)

    """ Set up Trainer"""

    training_args = TrainingArguments(
        output_dir="mlm_model",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNINGRATE,
        num_train_epochs=NUMOFEPOCH,
        load_best_model_at_end=True,
        weight_decay=WEIGHTDECAY,
        report_to="wandb",
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

    eval_results_validation = trainer.evaluate()
    print(f"Perplexity validation Split: {math.exp(eval_results_validation['eval_loss']):.2f}")

    eval_results_test = trainer.evaluate(eval_dataset=encoded_ds["test"])
    print(f"Perplexity test Split: {math.exp(eval_results_test['eval_loss']):.2f}")

    print("\n Inference:")

    # Inference
    text = "E-mail scam targets police chief Wiltshire Police warns about <mask> after its fraud squad chief was targeted."

    model.to("cpu")

    mask_filler = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    for i in mask_filler(text, top_k=5):
        print(i, "\n")

    wandb.finish()


