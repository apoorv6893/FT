# ============================================
# SIMPLE FINE-TUNING DEMO FOR GOOGLE COLAB
# WORKS FREE ON CPU/GPU
# ============================================

# Install required libraries
!pip -q install transformers datasets accelerate torch

# Imports
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from datasets import Dataset
import torch

# ============================================
# STEP 1: CREATE SMALL TRAINING DATA
# ============================================

training_data = [
    {
        "text": "User: My order is late\nBot: Sorry for the delay. Please share your order ID."
    },
    {
        "text": "User: I want refund\nBot: We can help process your refund request."
    },
    {
        "text": "User: Where is my package?\nBot: Please send your tracking number."
    },
    {
        "text": "User: Cancel my order\nBot: Sure. Please provide your order number."
    },
]

dataset = Dataset.from_list(training_data)

# ============================================
# STEP 2: LOAD SMALL MODEL
# ============================================

model_name = "distilgpt2"

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fix padding token
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

# ============================================
# STEP 3: TOKENIZE DATA
# ============================================

def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

    # Labels required for training
    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

tokenized_dataset = dataset.map(tokenize_function)

# ============================================
# STEP 4: BEFORE FINE-TUNING
# ============================================

print("\n==============================")
print("BEFORE FINE-TUNING")
print("==============================")

prompt = "User: My order is late\nBot:"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=20
)

before_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nMODEL OUTPUT:")
print(before_text)

# ============================================
# STEP 5: TRAINING CONFIGURATION
# ============================================

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=30,
    per_device_train_batch_size=2,
    logging_steps=1,
    save_steps=10,
    learning_rate=5e-5,
    fp16=False,
    report_to="none"
)

# ============================================
# STEP 6: DATA COLLATOR
# ============================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ============================================
# STEP 7: TRAINER
# ============================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# ============================================
# STEP 8: START TRAINING
# ============================================

print("\n==============================")
print("TRAINING STARTED")
print("==============================")

trainer.train()

# ============================================
# STEP 9: AFTER FINE-TUNING
# ============================================

print("\n==============================")
print("AFTER FINE-TUNING")
print("==============================")

outputs = model.generate(
    **inputs,
    max_new_tokens=20
)

after_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nMODEL OUTPUT:")
print(after_text)

# ============================================
# STEP 10: CLASSROOM EXPLANATION
# ============================================

print("\n==============================")
print("CLASSROOM SUMMARY")
print("==============================")

print("\n1. Base model gave generic responses")
print("2. We trained on custom examples")
print("3. Loss decreased during training")
print("4. Model learned customer-support style")
print("5. This process is called FINE-TUNING")
