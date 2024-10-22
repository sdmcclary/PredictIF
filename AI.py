import json
from datasets import Dataset, load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import torch

# 1. Load Data from JSON file
def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# Load your JSON file containing Python functions
# Assume each entry has a key "code" representing a Python function
json_file = 'your_dataset.json'  # Replace with your .json file path
data = load_data(json_file)

# Create a HuggingFace dataset from the JSON data
dataset = Dataset.from_dict({"code": [entry['code'] for entry in data]})

# 2. Preprocessing (Tokenization and Masking)
# Load a pre-trained tokenizer (You can use CodeT5 or another pre-trained model)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["code"], padding="max_length", truncation=True, max_length=512)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare the DataCollator for Masked Language Modeling (MLM) task
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15
)

# Split dataset for training and validation
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# 3. Pre-train the Model
# Load a pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
)

# Initialize the Trainer with model, tokenizer, datasets, and data_collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start Pre-training
trainer.train()

# Save the pre-trained model
trainer.save_model("./pretrained_model")

# 4. Fine-tuning: Mask 'if' statements in Python functions
# Modify Python functions to mask 'if' statements
def mask_if_statements(examples):
    masked_code = [code.replace('if', '<mask>') for code in examples['code']]
    return {"masked_code": masked_code}

# Apply masking to the original dataset
masked_dataset = dataset.map(mask_if_statements, batched=True)

# Tokenize the masked dataset
masked_tokenized_datasets = masked_dataset.map(tokenize_function, batched=True)

# Split into train, validation, and test sets (fine-tuning)
fine_tuning_split = masked_tokenized_datasets.train_test_split(test_size=0.2)
fine_tune_train_dataset = fine_tuning_split['train']
fine_tune_test_dataset = fine_tuning_split['test']

# 5. Fine-tune the pre-trained model on masked 'if' statements
# Load the pre-trained model for fine-tuning
fine_tuned_model = T5ForConditionalGeneration.from_pretrained("./pretrained_model")

# Set up training arguments for fine-tuning
fine_tune_args = TrainingArguments(
    output_dir="./fine_tuned_results",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=3,
)

# Initialize the Trainer for fine-tuning
fine_tune_trainer = Trainer(
    model=fine_tuned_model,
    args=fine_tune_args,
    train_dataset=fine_tune_train_dataset,
    eval_dataset=fine_tune_test_dataset,
    tokenizer=tokenizer,
)

# Start Fine-tuning
fine_tune_trainer.train()

# Save the fine-tuned model
fine_tune_trainer.save_model("./fine_tuned_model")

# 6. Evaluate on test set
eval_results = fine_tune_trainer.evaluate()

# Output the evaluation results
print("Evaluation Results:", eval_results)

