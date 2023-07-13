# Submission Exercise 1: MLM

1.) Install dependencies

`pip install -r requirements.txt`

2.) Login to wandb with your Key

`wandb login KEY`

3.) Run Python Script with the selected parameters
`python mlm.py BATCH_SIZE NUMOFEPOCH WEIGHTDECAY LEARNINGRATE`

The training command used to obtain the best model is:
`python mlm.py 32 10 0.005 1e-4`

Note: This script should be run on a GPU P100 with access > 15GB of memory.
