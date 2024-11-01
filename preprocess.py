import csv
import re

def cleanup(code):
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    code = "\n".join(line.strip() for line in code.splitlines())
    code = re.sub(r'\n\s*\n', '\n', code).strip()
    return code

def has_if_statement(code):
    return 'if ' in code

def mask_if_statements(code):
    match = re.search(r'(if\b[^\n]*:)', code)
    if match:
        target_block = match.group(0)
        masked_function = re.sub(r'if\b[^\n]*:', '<extra_id_0>', code, count=1)
    else:
        target_block = None
        masked_function = code
    return masked_function, target_block

def preprocess(in_path, out_path):
    with open(in_path, 'r', encoding="utf-8") as f:
        raw = f.read()

    lines = raw.splitlines()
    functions = []
    current_function = []

    for line in lines:
        if line.strip().startswith("def "):
            if current_function:
                functions.append(''.join(current_function))
                current_function = []  
            current_function.append(line)
        elif current_function:
            current_function.append(line)

    if current_function:
        functions.append(''.join(current_function))

    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['functions'])
        for funct in functions:
            cleaned_function = cleanup(funct)
            if has_if_statement(cleaned_function):
                writer.writerow([cleaned_function])

def split_datasets(input, pretraining, finetuning, masked):
    csv.field_size_limit(1000000)
    with open(input, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        functions = [row for row in reader]

    pretraining_data = functions[:127500]
    unmasked_data = functions[127500:150000]
    finetuning_data = functions[150000:200000]
    
    masked_data = []
    for unmasked in unmasked_data:
        masked_function, target_block = mask_if_statements(unmasked[0])
        masked_data.append([masked_function, target_block])

    with open(pretraining, 'w', newline='', encoding='utf-8') as pretrain_file:
        writer = csv.writer(pretrain_file)
        writer.writerow(['pretraining'])
        writer.writerows(pretraining_data)

    with open(masked, 'w', newline='', encoding='utf-8') as masked_file:
        writer = csv.writer(masked_file)
        writer.writerow(['Masked', 'target_block'])  # Header with target_block
        for data in masked_data:
            writer.writerow(data)  # Write the masked function and target_block

    with open(finetuning, 'w', newline='', encoding='utf-8') as finetune_file:
        writer = csv.writer(finetune_file)
        writer.writerow(['finetuning'])
        writer.writerows(finetuning_data)

# Paths to files
in_path = 'python_functions.txt'
out_path = 'functions.csv'
pretraining_dataset = 'pretraining_dataset.csv'
finetuning_dataset = 'fine_tuning_dataset.csv'
pretraining_masked_dataset = 'pretraining_masked.csv'

preprocess(in_path, out_path)
split_datasets(out_path, pretraining_dataset, finetuning_dataset, pretraining_masked_dataset)
