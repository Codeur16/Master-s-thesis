import os
import time
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from datasets import Dataset
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)

# === 1. Config ===
MODEL_NAME = "bert-large-uncased"
OUTPUT_DIR = "./bert-large-nli-v2/"
LOG_DIR = "./logs"
DATASET_FILE = "/home/ingenieur/Desktop/Memoire de master2/Experimantations-on-multi_nli/data/balanced_10k_dataset_v2.jsonl"

# Configuration parameters - matching project specifications
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3

def load_nli_dataset(file_path):
    """
    Charge le dataset NLI depuis le fichier JSONL
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading dataset"):
            data.append(json.loads(line.strip()))
    
    df = pd.DataFrame(data)
    return Dataset.from_pandas(df)

def get_label_name(label):
    """
    Convertit le label numérique en texte
    """
    if label == 0:
        return "entailment"
    elif label == 1:
        return "neutral"
    elif label == 2:
        return "contradiction"
    else:
        return "unknown"

# === 2. Load dataset ===
print("Loading dataset...")
dataset = load_nli_dataset(DATASET_FILE)
print(f"Total dataset size: {len(dataset)}")

# Split dataset
train_dataset = dataset.train_test_split(test_size=0.2, seed=42)['train']
val_dataset = dataset.train_test_split(test_size=0.2, seed=42)['test']

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")

# === 3. Load tokenizer ===
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# === 4. Preprocessing ===
def preprocess_function(examples):
    """
    Préprocesse les données pour BERT
    """
    # Filter out duplicate examples (e.g., "p == h")
    premises = examples['premise']
    hypotheses = examples['hypothesis']
    labels = examples['label']
    
    clean_data = [(p, h, l) for p, h, l in zip(premises, hypotheses, labels) 
                  if p.strip().lower() != h.strip().lower()]
    
    if not clean_data:
        return {}
    
    clean_premises, clean_hypotheses, clean_labels = zip(*clean_data)
    
    # Tokenize premise and hypothesis pairs
    model_inputs = tokenizer(
        list(clean_premises), 
        list(clean_hypotheses),
        truncation=True, 
        padding="max_length", 
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    model_inputs["labels"] = torch.tensor(clean_labels, dtype=torch.long)
    return model_inputs

print("Preprocessing datasets...")
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train dataset"
)

tokenized_val = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing validation dataset"
)

# === 5. Load model ===
print("Loading model...")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# === 6. Metrics ===
def compute_metrics(eval_pred):
    """
    Calcule les métriques pour l'évaluation
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    acc = np.mean([1 if p == l else 0 for p, l in zip(predictions, labels)])
    
    # Calculate F1 score
    f1 = f1_score(labels, predictions, average='macro')
    
    return {
        "accuracy": acc,
        "f1_score": f1
    }

# === 7. Training arguments ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    logging_strategy="steps",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=2e-5,  # BERT typical learning rate
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir=LOG_DIR,
    load_best_model_at_end=True,
    report_to="tensorboard",
    fp16=torch.cuda.is_available(),
    dataloader_drop_last=True,
    remove_unused_columns=False
)

# === 8. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# === 9. Fine-tuning ===
print("Starting BERT fine-tuning...")
start_time = time.time()
trainer.train()
end_time = time.time()

print("Training completed!")

# === 10. Save model ===
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# === 11. Evaluation ===
print("Final evaluation...")
results = trainer.evaluate()
print(f"\nEvaluation results: {results}")

# === 12. Save metrics & training time ===
metrics = {
    "training_time": end_time - start_time,
    "eval_accuracy": results.get("eval_accuracy", 0.0),
    "eval_f1_score": results.get("eval_f1_score", 0.0)
}

with open(os.path.join(OUTPUT_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
    
print(f"\nTraining completed in {metrics['training_time']:.2f} seconds")
print(f"Model saved to {OUTPUT_DIR}")