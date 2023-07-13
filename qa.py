import random
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import TrainingArguments
import wandb
import sys

# Load the QASPER dataset
dataset = load_dataset("allenai/qasper")

# Print the first sample from the train split
# print(dataset['train'][0])

BATCH_SIZE = int(sys.argv[1])
NUMOFEPOCH = int(sys.argv[2])
WEIGHTDECAY = float(sys.argv[3])
LEARNINGRATE = float(sys.argv[4])

def printArgs():
    print("Training Arguments: \n")
    print("Batch size: ", BATCH_SIZE, "typ: ", type(BATCH_SIZE))
    print("Num of Epoch: ", NUMOFEPOCH, "typ: ", type(NUMOFEPOCH))
    print("Weight Decay: ", WEIGHTDECAY, "typ: ", type(WEIGHTDECAY))
    print("Lerning Rate: ", LEARNINGRATE, "typ: ", type(LEARNINGRATE))

printArgs()

wandb.init(
    project="qasper-project",
    config={
    "learning_rate": LEARNINGRATE,
    "architecture": "Google-T5",
    "dataset": "QASPER",
    "epochs": NUMOFEPOCH,
    }
)

# Perform EDA
def analyze_split(train_data):
    num_questions_per_article = []
    num_answers_per_question = []
    num_free_form_answers = 0
    num_extractive_answers = 0
    num_unanswerable_questions = 0
    abstract_lengths = []
    abstract_word_counts = []
    answer_word_counts = []

    for instance in train_data:
        # Count number of questions per article
        num_questions_per_article.append(len(instance['qas']['question']))

        for answer_list in instance['qas']['answers']:
            # Count number of answers per question
            num_answers_per_question.append(len(answer_list['answer']))

            for answer in answer_list['answer']:
                # Check answer type
                # if 'free_form_answer' in answer:
                if answer['free_form_answer'] != '':
                    num_free_form_answers += 1
                    answer_word_counts.append(len(answer['free_form_answer'].split(" ")))
                if answer['extractive_spans'] != '':
                    num_extractive_answers += 1
                    
                    if type(answer["extractive_spans"]) == list:
                        answer_word_counts.append(len(", ".join(answer['extractive_spans']).split(" ")))
                    else:
                        answer_word_counts.append(len(answer['extractive_spans'].split(" ")))
                if answer['unanswerable']:
                    num_unanswerable_questions += 1

        # Compute abstract length
        abstract_lengths.append(len(instance['abstract']))
        abstract_word_counts.append(len(instance['abstract'].split(" ")))

    # Distribution of number of questions per article
    question_counts = {}
    for count in num_questions_per_article:
        question_counts[count] = question_counts.get(count, 0) + 1

    # Average number of answers per question
    avg_answers_per_question = sum(num_answers_per_question) / len(num_answers_per_question)

    # Print the results
    print("Distribution of Number of Questions per Article:")
    for count, freq in sorted(question_counts.items(), key=lambda x: x[0]):
        print(f"{count} questions: {freq} articles")

    print("Average Number of Answers per Question:", avg_answers_per_question)
    print("Number of Free-form Answers:", num_free_form_answers)
    print("Number of Extractive Answers:", num_extractive_answers)
    print("Number of Unanswerable Questions:", num_unanswerable_questions)

    # Distribution of abstract lengths
    print("Distribution of Abstract Lengths:")
    print("Minimum length:", min(abstract_lengths))
    print("Maximum length:", max(abstract_lengths))
    print("Mean length:", sum(abstract_lengths) / len(abstract_lengths))
    print("Distribution of Abstract Word Counts:")
    print("Minimum word counts:", min(abstract_word_counts))
    print("Maximum word counts:", max(abstract_word_counts))
    print("Mean word counts:", sum(abstract_word_counts) / len(abstract_word_counts))
    print("Distribution of Answer Word Counts:")
    print("Minimum word counts:", min(answer_word_counts))
    print("Maximum word counts:", max(answer_word_counts))
    print("Mean word counts:", sum(answer_word_counts) / len(answer_word_counts))

# EDA for train and test splits
print("\nTrain Split:")
analyze_split(dataset['train'])
print("\nTest Split:")
analyze_split(dataset['test'])

# Flattening the hierarchical structure
def flatten_split(split):
    count = 0
    abstracts, questions, answers = [], [], []
    for sample in split:
        for i in range(len(sample['qas']['question'])):
            abstracts.append(sample['abstract'])
            questions.append(sample['qas']['question'][i])
            if len(sample['qas']['answers'][i]['answer']) > 0:
                answer = random.choice(sample['qas']['answers'][i]['answer'])
                if answer['unanswerable']:
                    answers.append('')
                elif answer['free_form_answer'] != '':
                    answers.append(answer["free_form_answer"])
                elif answer['extractive_spans'] != '':
                    if type(answer["extractive_spans"]) == list:
                        answers.append(", ".join(answer["extractive_spans"]))
                    else:
                        answers.append(answer['extractive_spans'])
            else:
                answers.append('')
    return abstracts, questions, answers

train_abstracts, train_questions, train_answers = flatten_split(dataset['train'])
test_abstracts, test_questions, test_answers = flatten_split(dataset['test'])

train = {'abstract': train_abstracts, 'question': train_questions, 'answer': train_answers}

test = {'abstract': test_abstracts, 'question': test_questions, 'answer': test_answers}

train_dataset = Dataset.from_dict(train)
test_dataset = Dataset.from_dict(test)

flattened_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

assert len(train_abstracts) == len(train_questions) == len(train_answers)

# Split train dataset into train and validation sets
data_split = train_dataset.train_test_split(test_size=0.1)
train_dataset = data_split["train"]
validation_dataset = data_split["test"]

# Define preprocessing function
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def preprocess(data):
    input_text = ['question: ' + question + ' context: ' + abstract for (question, abstract) in zip(data["question"], data['abstract'])]
    target_text = data['answer']
    tokenized_inputs = tokenizer(input_text, truncation=True, max_length=128, padding='max_length')
    tokenized_targets = tokenizer(target_text, truncation=True, max_length=32, padding='max_length')
    return {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': tokenized_targets['input_ids']
    }

# Apply the preprocessing function
train_dataset = train_dataset.map(
    preprocess,
    batched=True,
    remove_columns=train_dataset.column_names
)

valid_dataset = validation_dataset.map(
    preprocess,
    batched=True,
    remove_columns=validation_dataset.column_names
)

# Load the model
# model = T5ForConditionalGeneration.from_pretrained('google/t5-efficient-tiny') 
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Set up Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="qa_model",
    per_device_train_batch_size=BATCH_SIZE,  
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNINGRATE,
    weight_decay=WEIGHTDECAY,
    load_best_model_at_end=True,
    num_train_epochs=NUMOFEPOCH,  # adjust number of epochs as per requirement
    report_to="wandb"
)

# Set up Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Training
trainer.train()

wandb.finish()

# Predict new sample from test set
from transformers import T5Tokenizer

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

model.to("cpu")

# Function to generate an answer to a question given the context
def answer_question(question, context):
    # Encode the question and context with the T5 tokenizer
    # T5 expects a specific format of text, which is "question: some question context: some context"
    encoded_input = tokenizer.encode("question: " + question + " context: " + context, return_tensors='pt')

    # Generate the answer with the model
    output = model.generate(encoded_input, max_length=512, num_beams=4, early_stopping=True)

    # Decode the output tensor to a string
    answer = tokenizer.decode(output[0])
    return answer

id = 893
print("Question:", test_dataset[id]["question"])
print("Predicted Answer: ", answer_question(test_dataset[id]["question"], test_dataset[id]["abstract"]))
print("Context:", test_dataset[id]["abstract"])
print("Answer:", test_dataset[id]["answer"])