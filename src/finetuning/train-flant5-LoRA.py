# flant5 finetubed v2
import os
import time
import json
import torch
import pandas as pd
import numpy as np
import evaluate
from tqdm import tqdm
from sklearn.metrics import f1_score
from datasets import Dataset
from accelerate import Accelerator
from transformers import (
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# === 1. Config ===
MODEL_NAME = "/kaggle/input/flan-t5/pytorch/xl/3"
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 8
BATCH_SIZE = 16
EPOCHS = 3
USE_KBIT = True
LOG_DIR = "./logs"
OUTPUT_DIR = ".//kaggle/input/flan-t5/pytorch/xl/3-lora-MULTI-NLI-V2-850M"
TRAIN_FILE = "/kaggle/input/multinli-textual-entailment-corpus/train.csv"
VAL_FILE = "/kaggle/input/multinli-textual-entailment-corpus/validation_matched.csv"

accelerator = Accelerator()
device = accelerator.device

# === 2. Load tokenizer and base model ===
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
base_model.gradient_checkpointing_enable()
base_model.to(device)

# === 3. Optionally prepare model for k-bit training ===
if USE_KBIT:
    base_model = prepare_model_for_kbit_training(base_model)

# === 4. Apply LoRA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "k"], #, "v", "o", "wi", "wo"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# === 5. Load dataset ===
def load_data_to_dataset(file_path):
    df = pd.read_csv(file_path)
    return Dataset.from_pandas(df)

train_dataset1 = load_data_to_dataset(TRAIN_FILE)
val_dataset1 = load_data_to_dataset(VAL_FILE)

train_dataset = train_dataset1.train_test_split(test_size=0.2, seed=42)['train']
val_dataset = val_dataset1.train_test_split(test_size=0.2, seed=42)['train']

# === 6. Preprocessing ===
def get_label_name(label):
    if label == 0:
        return "entailment"
    elif label == 1:
        return "neutral"
    elif label == 2:
        return "contradiction"
    else:
        return "unknown"

def preprocess_function(examples):
    inputs = [f"nli premise: {p} hypothesis: {h}" for p, h in zip(examples['premise'], examples['hypothesis'])]
    labels_text = examples['label']     
    targets = [get_label_name(label) for label in labels_text]

    # Filter out duplicate examples (e.g., "p == h")
    clean_data = [(inp, tgt) for inp, tgt in zip(inputs, targets) if inp.strip().lower() != tgt.strip().lower()]
    if not clean_data:
        return {}

    model_inputs = tokenizer(
        [x[0] for x in clean_data], 
        max_length=MAX_INPUT_LENGTH, 
        truncation=True, 
        padding="max_length"
    )
    
    labels = tokenizer(
        [x[1] for x in clean_data],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    ).input_ids

    model_inputs["labels"] = labels
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

# === 7. Metrics ===
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    acc = np.mean([1 if p == l else 0 for p, l in zip(decoded_preds, decoded_labels)])
    
    label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    y_true = [label_map.get(l, -1) for l in decoded_labels]
    y_pred = [label_map.get(p, -1) for p in decoded_preds]
    
    valid_idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != -1 and p != -1]
    y_true = [y_true[i] for i in valid_idx]
    y_pred = [y_pred[i] for i in valid_idx]
    
    f1 = f1_score(y_true, y_pred, average='macro') if valid_idx else 0.0
    
    return {
        "accuracy": acc,
        "f1_score": f1
    }

# === 8. Training arguments ===
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    logging_strategy="steps",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=2e-4,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir=LOG_DIR,
    load_best_model_at_end=True,
    report_to="tensorboard",
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    label_smoothing_factor=0.1,
)

# === 9. Trainer ===
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# === 10. Fine-tuning ===
print("Starting LoRA fine-tuning...")
start_time = time.time()
trainer.train()
end_time = time.time()

# === 11. Save only LoRA adapters ===
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# === 12. Evaluation ===
results = trainer.evaluate()
print(f"\nEvaluation results: {results}")

# === 13. Save metrics & training time ===
metrics = {
    "training_time": end_time - start_time,
    "eval_accuracy": results.get("eval_accuracy", 0.0),
    "eval_f1_score": results.get("eval_f1_score", 0.0)
}

with open(os.path.join(OUTPUT_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
    
print(f"\nTraining completed in {metrics['training_time']:.2f} seconds")
print(f"Model saved to {OUTPUT_DIR}")
