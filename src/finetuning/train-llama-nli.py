import os
import time
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import evaluate
from tqdm import tqdm
from sklearn.metrics import f1_score
from datasets import Dataset
from accelerate import Accelerator
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# === 1. Config ===
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B"
OUTPUT_DIR = "./finetuned-llama-nli-v2/"
LOG_DIR = "./logs"
DATASET_FILE = "/home/ingenieur/Desktop/Memoire de master2/Experimantations-on-multi_nli/data/balanced_10k_dataset_v2.jsonl"

# NLI-specific configuration -
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 8  # For classification output
BATCH_SIZE = 16
EPOCHS = 3
USE_KBIT = True

accelerator = Accelerator()
device = accelerator.device

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params
    print(f"Trainable Params: {trainable_params} || Total Params: {total_params} || Trainable %: {trainable_percent:.2f}%")

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

def format_nli_prompt(premise, hypothesis):
    """
    Formate l'entrée pour la tâche NLI avec Llama
    """
    prompt = f"nli premise: {premise} hypothesis: {hypothesis}"
    return prompt

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

def preprocess_function(examples, tokenizer):
    """
    Préprocesse les données pour le fine-tuning
    """
    inputs = [format_nli_prompt(p, h) for p, h in zip(examples['premise'], examples['hypothesis'])]
    labels_text = examples['label']     
    targets = [get_label_name(label) for label in labels_text]
    
    # Filter out duplicate examples (e.g., "p == h")
    clean_data = [(inp, tgt) for inp, tgt in zip(inputs, targets) if inp.strip().lower() != tgt.strip().lower()]
    if not clean_data:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    # For causal LM, we create full text with prompt + answer
    full_texts = []
    for inp, tgt in clean_data:
        full_text = f"{inp}\nAnswer: {tgt}{tokenizer.eos_token}"
        full_texts.append(full_text)
    
    # Tokenize full texts
    model_inputs = tokenizer(
        full_texts,
        max_length=MAX_INPUT_LENGTH + MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors=None  # Changed from "pt" to None for compatibility
    )
    
    # For causal LM, labels are the same as input_ids
    model_inputs["labels"] = [ids.copy() for ids in model_inputs["input_ids"]]
    
    # Mask the prompt part in labels (only compute loss on the answer)
    for i, (inp, full_text) in enumerate(zip([x[0] for x in clean_data], full_texts)):
        prompt_with_answer_prefix = f"{inp}\nAnswer: "
        prompt_length = len(tokenizer.encode(prompt_with_answer_prefix, add_special_tokens=False))
        # Ensure we don't exceed the sequence length
        if prompt_length < len(model_inputs["labels"][i]):
            model_inputs["labels"][i] = [-100] * prompt_length + model_inputs["labels"][i][prompt_length:]
        else:
            model_inputs["labels"][i] = [-100] * len(model_inputs["labels"][i])
    
    return model_inputs

# === 2. Load Model and Tokenizer ===
# Quantization Configuration for Efficient Loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

print("Loading model...")
# Load Model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Enable Gradient Checkpointing for Memory Efficiency
model.gradient_checkpointing_enable()

# === 3. Optionally prepare model for k-bit training ===
if USE_KBIT:
    model = prepare_model_for_kbit_training(model)

# Enables Gradients for Input Embeddings
model.enable_input_require_grads()

# Configure LoRA
class CastOutputToFloat(nn.Sequential):
    """
    Ensures output of the model is cast to bfloat16.
    """
    def forward(self, x):
        return super().forward(x).to(torch.bfloat16)

# Apply Casting to output layer
model.lm_head = CastOutputToFloat(model.lm_head)

# === 4. Apply LoRA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap Model with LoRA
lora_model = get_peft_model(model, lora_config)

# Print Trainable Parameter Info
lora_model.print_trainable_parameters()

# Adjust Model Configuration
lora_model.config.use_cache = False
lora_model.config.pad_token_id = tokenizer.pad_token_id

# === 5. Load dataset ===
print("Loading dataset...")
dataset = load_nli_dataset(DATASET_FILE)
print(f"Total dataset size: {len(dataset)}")

# Split dataset
train_dataset = dataset.train_test_split(test_size=0.2, seed=42)['train']
val_dataset = dataset.train_test_split(test_size=0.2, seed=42)['test']

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")

# === 6. Preprocessing ===
print("Preprocessing datasets...")
tokenized_train = train_dataset.map(
    lambda x: preprocess_function(x, tokenizer),
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train dataset"
)

tokenized_val = val_dataset.map(
    lambda x: preprocess_function(x, tokenizer),
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing validation dataset"
)

# === 7. Metrics ===
def compute_metrics(eval_pred):
    """
    Calcule les métriques pour l'évaluation
    """
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    decoded_preds = []
    decoded_labels = []
    
    # Convert to numpy arrays if they aren't already
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Get the predicted token IDs (argmax for each position)
    if len(predictions.shape) == 3:  # (batch_size, seq_len, vocab_size)
        pred_ids = np.argmax(predictions, axis=-1)
    else:
        pred_ids = predictions
    
    for pred, label in zip(pred_ids, labels):
        try:
            # Take only non-masked tokens for evaluation
            valid_label_mask = label != -100
            if not np.any(valid_label_mask):
                continue
                
            valid_labels = label[valid_label_mask]
            valid_preds = pred[valid_label_mask] if len(pred) >= len(valid_labels) else pred
            
            # Ensure same length
            min_len = min(len(valid_labels), len(valid_preds))
            valid_labels = valid_labels[:min_len]
            valid_preds = valid_preds[:min_len]
            
            # Decode
            pred_text = tokenizer.decode(valid_preds, skip_special_tokens=True)
            label_text = tokenizer.decode(valid_labels, skip_special_tokens=True)
            
            # Extract predicted label (last word after "Answer:")
            pred_label = extract_predicted_label(pred_text)
            true_label = extract_predicted_label(label_text)
            
            decoded_preds.append(pred_label)
            decoded_labels.append(true_label)
        except Exception as e:
            print(f"Warning: Error processing prediction: {e}")
            continue
    
    if not decoded_preds or not decoded_labels:
        return {"accuracy": 0.0, "f1_score": 0.0}
    
    # Calculate accuracy
    acc = np.mean([1 if p == l else 0 for p, l in zip(decoded_preds, decoded_labels)])
    
    # Calculate F1
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

def extract_predicted_label(text):
    """
    Extrait le label prédit du texte généré
    """
    text = text.lower().strip()
    if "entailment" in text:
        return "entailment"
    elif "neutral" in text:
        return "neutral"
    elif "contradiction" in text:
        return "contradiction"
    else:
        return "unknown"

# === 8. Training arguments ===
training_args = TrainingArguments(
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
    fp16=torch.cuda.is_available(),
    label_smoothing_factor=0.1,
    dataloader_drop_last=True,
    remove_unused_columns=False,
    prediction_loss_only=False,  # Add this for proper evaluation
    include_inputs_for_metrics=True  # Add this for better metrics handling
)

# === 9. Data Collator ===
data_collator = transformers.DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False,
    pad_to_multiple_of=8
)

# === 10. Trainer ===
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# === 11. Fine-tuning ===
print("Starting LoRA fine-tuning...")
start_time = time.time()
trainer.train()
end_time = time.time()

print("Training completed!")

# === 12. Save only LoRA adapters ===
print("Saving model...")
try:
    lora_model.save_pretrained(OUTPUT_DIR, safe_serialization=False)
    tokenizer.save_pretrained(OUTPUT_DIR, safe_serialization=False)
    print(f"✅ Model successfully saved to {OUTPUT_DIR}")
except Exception as e:
    print(f"❌ Error saving model: {e}")
    print("Attempting fallback save...")
    try:
        lora_model.save_pretrained(OUTPUT_DIR + "_fallback", safe_serialization=False)
        tokenizer.save_pretrained(OUTPUT_DIR + "_fallback", safe_serialization=False)
        print(f"✅ Model saved to fallback location: {OUTPUT_DIR}_fallback")
    except Exception as e2:
        print(f"❌ Fallback save also failed: {e2}")

# === 13. Evaluation ===
print("Final evaluation...")
try:
    results = trainer.evaluate()
    print(f"\nEvaluation results: {results}")
except Exception as e:
    print(f"❌ Error during evaluation: {e}")
    results = {"eval_accuracy": 0.0, "eval_f1_score": 0.0}

# === 14. Save metrics & training time ===
metrics = {
    "training_time": end_time - start_time,
    "eval_accuracy": results.get("eval_accuracy", 0.0),
    "eval_f1_score": results.get("eval_f1_score", 0.0)
}

try:
    with open(os.path.join(OUTPUT_DIR, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved successfully")
except Exception as e:
    print(f"❌ Error saving metrics: {e}")
    
print(f"\nTraining completed in {metrics['training_time']:.2f} seconds")
print(f"Model saved to {OUTPUT_DIR}") 