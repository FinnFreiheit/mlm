import torch
from transformers import AutoConfig, AutoModelForMaskedLM
from const import DEBUG, ps

if __name__ == "__main__":

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
