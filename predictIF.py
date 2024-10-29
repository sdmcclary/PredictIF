import json
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments
)
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 1. Load a Public Dataset (CodeSearchNet - Python subset)
dataset = load_dataset("code_search_net", "python", split="train[:1%]")  # Using a small subset for testing
print(dataset[0])
# 2. Preprocessing (Tokenization and Custom Masking)
# Load a pre-trained tokenizer
#tokenizer = T5Tokenizer.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
print(tokenizer.encode("<extra_id_0>"))
# Custom masking function: Replace 'if' statements with <extra_id_0>
def mask_if_statements(examples):
    masked_code = [code.replace('if', '<extra_id_0>') for code in examples['func_code_string']]
    return {"input_text": masked_code, "target_text": examples['func_code_string']}

# Apply masking to create input-output pairs for T5
masked_dataset = dataset.map(mask_if_statements, batched=True)

# Tokenization function for T5 input-output format
def tokenize_function(examples):
    model_inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=512).input_ids
    model_inputs["labels"] = labels
    return model_inputs

# Tokenize the dataset
tokenized_datasets = masked_dataset.map(tokenize_function, batched=True)

# Split into train and test sets for training
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# 3. Training: Prepare T5 for Conditional Generation
# Load a pre-trained T5 model
#model = T5ForConditionalGeneration.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
model.to("cuda")
# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,  # Reduced for testing
    weight_decay=0.01,
    save_total_limit=3,
)

# Initialize the Trainer with model, tokenizer, and datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Start Training
trainer.train()

# Save the trained model
trainer.save_model("./trained_model")

# 4. Evaluation on test set
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# 5. Test Example
# Use the trained model to predict 'if' statements in a sample code snippet
test_code = "x = 10\n<extra_id_0> x > 5:\n    print('Greater than 5')\nelse:\n    print('5 or less')"
input_ids = tokenizer(test_code, return_tensors="pt").input_ids.to("cuda")

# Generate prediction
output_ids = model.generate(input_ids,do_sample=True)
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Masked Code:", test_code)
print("Model Prediction:", output)

