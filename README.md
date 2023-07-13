# Run in Kaggle

1.) Clone Repo in Kaggle-Notebook

`!git clone https://github.com/FinnFreiheit/mlm.git`

2.) Install Dependencies

`!pip install -r /kaggle/working/mlm/requirements.txt`

3.) Restart the run-time

4.) Login to wandb with your Key

`!wandb login KEY`

## Exercise 1: Masked language modeling Task

Run Python Script with the selected parameters
`!python /kaggle/working/mlm/mlm.py BATCH_SIZE NUMOFEPOCH WEIGHTDECAY LEARNINGRATE`

For example
`!python /kaggle/working/mlm/mlm.py 32 3 0.01 2e-5`

## Exercise 2: Q&A Task

Run Python Script with the selected parameters
`!python /kaggle/working/mlm/qa.py BATCH_SIZE NUMOFEPOCH WEIGHTDECAY LEARNINGRATE`

For example
`!python /kaggle/working/mlm/qa.py 32 8 0.01 1e-3`
