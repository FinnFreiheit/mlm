from data import data_collator, encoded_ds
from model import model
from transformers import TrainingArguments, Trainer

if __name__ == '__main__':

    dropout_probability = 0.15
    # model.roberta.encoder.layer[i].output.dropout = ?

    # Inference
    text = "E-mail scam targets police chief Wiltshire Police warns about <mask> after its fraud squad chief was targeted."

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
