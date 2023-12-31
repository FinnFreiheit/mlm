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

Our final parameters are:
`!python /kaggle/working/mlm/mlm.py 32 3 0.01 2e-5`

## Exercise 2: Q&A Task

Run Python Script with the selected parameters
`!python /kaggle/working/mlm/qa.py BATCH_SIZE NUMOFEPOCH WEIGHTDECAY LEARNINGRATE`

For example
`!python /kaggle/working/mlm/qa.py 32 10 0.005 1e-4`

Note: This script should be run on a GPU P100 with access > 15GB of memory.
