import os
import time
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score
# === 1. Config ===
BASE_MODEL_NAME = "/kaggle/input/mistral/pytorch/7b-instruct-v0.1-hf/1"
OUTPUT_DIR = "./finetuned-mistral-nli/"
DATASET_FILE = "/home/ingenieur/Desktop/Memoire de master2/Experimantations-on-multi_nli/data/balanced_10k_dataset_v2_enriched.csv"

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 8
BATCH_SIZE = 4  # réduit pour Kaggle GPU
EPOCHS = 3
USE_KBIT = True

accelerator = Accelerator()
device = accelerator.device

# === 2. Dataset ===
df = pd.read_csv(DATASET_FILE)
dataset = Dataset.from_pandas(df)
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split['train']
val_dataset = split['test']

# === 3. Tokenizer & preprocessing ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def format_nli_prompt(premise, hypothesis):
    return f"nli premise: {premise} hypothesis: {hypothesis}"

def get_label_name(label):
    return {0:"entailment",1:"neutral",2:"contradiction"}.get(label,"unknown")

def preprocess_function(examples):
    inputs = [format_nli_prompt(p,h) for p,h in zip(examples['premise'], examples['hypothesis'])]
    targets = [get_label_name(l) for l in examples['label']]
    full_texts = [f"{inp}\nAnswer: {t}{tokenizer.eos_token}" for inp,t in zip(inputs,targets)]
    model_inputs = tokenizer(
        full_texts,
        max_length=MAX_INPUT_LENGTH + MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )
    model_inputs["labels"] = [ids.copy() for ids in model_inputs["input_ids"]]
    # mask prompt
    for i, inp in enumerate(inputs):
        prefix_len = len(tokenizer.encode(f"{inp}\nAnswer: ", add_special_tokens=False))
        if prefix_len < len(model_inputs["labels"][i]):
            model_inputs["labels"][i] = [-100]*prefix_len + model_inputs["labels"][i][prefix_len:]
        else:
            model_inputs["labels"][i] = [-100]*len(model_inputs["labels"][i])
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

# === 4. Model loading with 4-bit quantization ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
     device_map={"": 0},
    torch_dtype=torch.bfloat16
)

model.gradient_checkpointing_enable()
if USE_KBIT:
    model = prepare_model_for_kbit_training(model)
model.enable_input_require_grads()

# === 5. LoRA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","k_proj"],#,"v_proj","o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
lora_model.config.use_cache = False
lora_model.config.pad_token_id = tokenizer.pad_token_id

# === 6. Data collator ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

# === 7. Metrics ===
def extract_predicted_label(text):
    text = text.lower().strip()
    if "entailment" in text:
        return "entailment"
    elif "neutral" in text:
        return "neutral"
    elif "contradiction" in text:
        return "contradiction"
    else:
        return "unknown"

def compute_metrics(eval_pred):
    """
    Calcule accuracy et F1 pour l'évaluation NLI.
    """
    predictions, labels = eval_pred
    decoded_preds, decoded_labels = [], []

    # S'assurer que les tenseurs sont sur CPU pour numpy
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    # Si forme (batch, seq_len, vocab_size)
    if len(predictions.shape) == 3:
        pred_ids = np.argmax(predictions, axis=-1)
    else:
        pred_ids = predictions

    for pred, label in zip(pred_ids, labels):
        valid_mask = label != -100
        if not np.any(valid_mask):
            continue
        valid_labels = label[valid_mask]
        valid_preds = pred[valid_mask] if len(pred) >= len(valid_labels) else pred
        min_len = min(len(valid_labels), len(valid_preds))
        valid_labels = valid_labels[:min_len]
        valid_preds = valid_preds[:min_len]

        pred_text = tokenizer.decode(valid_preds, skip_special_tokens=True)
        label_text = tokenizer.decode(valid_labels, skip_special_tokens=True)

        decoded_preds.append(extract_predicted_label(pred_text))
        decoded_labels.append(extract_predicted_label(label_text))

    if not decoded_preds or not decoded_labels:
        return {"accuracy": 0.0, "f1_score": 0.0}

    # Accuracy simple
    acc = np.mean([1 if p == l else 0 for p, l in zip(decoded_preds, decoded_labels)])

    # F1 macro
    label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    y_true = [label_map.get(l, -1) for l in decoded_labels]
    y_pred = [label_map.get(p, -1) for p in decoded_preds]

    valid_idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != -1 and p != -1]
    if not valid_idx:
        return {"accuracy": acc, "f1_score": 0.0}

    y_true = [y_true[i] for i in valid_idx]
    y_pred = [y_pred[i] for i in valid_idx]

    f1 = f1_score(y_true, y_pred, average='macro') if len(set(y_true)) > 1 else 0.0

    return {
        "accuracy": acc,
        "f1_score": f1
    }




# === 8. Training arguments ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    learning_rate=2e-4,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    fp16=True,
    gradient_accumulation_steps=4,  
    report_to="tensorboard",
    dataloader_drop_last=True,
    remove_unused_columns=False,
    metric_for_best_model="f1_score",   
    greater_is_better=True               
)

# === 9. Trainer ===
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# === 10. Fine-tuning ===
print("Starting LoRA fine-tuning...")
start_time = time.time()
trainer.train()
end_time = time.time()
print("Training completed!")
# === 10. Save LoRA adapters only ===
print("Saving model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
lora_model.save_pretrained(OUTPUT_DIR)
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
