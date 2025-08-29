""" ----------- NeuroPEFT -------------- """

import os
import time
import json
import torch
import numpy as np
import pandas as pd
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
    EarlyStoppingCallback,
    EvalPrediction
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


""" --------- SymbolicRules Class --------- """

class SymbolicRules:
    """Classe contenant toutes les règles heuristiques pour l'analyse textuelle"""
    
    # ===== Règles pour paires texte (premise/hypothesis) =====
    
    @staticmethod
    def lexical_overlap_rule(premise, hypothesis):
        """Règle 1 : Recouvrement lexical"""
        stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
        'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can',
        'could', 'you', 'your', 'yours', 'i', 'me', 'my', 'mine', 'he', 'him',
        'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their',
        'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
        'whose', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'
        }
        def get_content_words(text):
            return [word.lower() for word in text.split() 
                    if word.lower() not in stop_words]
        premise_words = get_content_words(premise)
        hypothesis_words = get_content_words(hypothesis)
    
        common_words = set(premise_words) & set(hypothesis_words)
        return len(common_words) > 1

    @staticmethod
    def exact_substring_rule(premise, hypothesis):
        """Règle 2 : Inclusion exacte"""
        return hypothesis.strip().lower() in premise.strip().lower()

    @staticmethod
    def negation_rule(premise, hypothesis):
        """
        Enhanced negation detection that handles:
        1. Explicit negation words
        2. Negative prefixes
        3. Negative contractions
        4. Double negatives
        5. Implicit negations
        """
        # Expanded set of negation words
        negation_words = {
            # Basic negations
            "no", "not", "never", "nothing", "none", "nobody", "neither",
            # Additional negations
            "nowhere", "nor", "without", "lack", "deny", "refuse", "reject",
            # Negative pronouns
            "nobody", "nothing", "none", "no one", "neither",
            # Negative adverbs
            # "hardly", "barely", "scarcely", "rarely",
            # Prepositions implying negation
            # "against", "despite", "except", "unless",
        }
        
        # Negative contractions
        negative_contractions = {
            "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
            "hadn't", "won't", "wouldn't", "don't", "doesn't", "didn't",
            "can't", "couldn't", "shouldn't", "mustn't", "ain't"
        }
        
        # Negative prefixes to check
        negative_prefixes = ("un", "in", "im", "il", "ir", "dis", "mis", "non", "anti")
        
        def contains_negative_prefix(word):
            """Check if word contains negative prefix."""
            return any(word.startswith(prefix) and len(word) > len(prefix) + 1 
                    for prefix in negative_prefixes)
        
        def tokenize(text):
            """Simple tokenization while preserving contractions."""
            # Convert to lowercase and split
            return text.lower().split()
        
        def has_negation(text):
            tokens = tokenize(text)
            
            # Check for explicit negation words and contractions
            if any(token in negation_words for token in tokens):
                return True
            if any(token in negative_contractions for token in tokens):
                return True
            
            # Check for words with negative prefixes
            if any(contains_negative_prefix(token) for token in tokens):
                return True
            
            return False
    
        # Check both premise and hypothesis for negations
        premise_has_negation = has_negation(premise)
        hypothesis_has_negation = has_negation(hypothesis)
        
        return premise_has_negation != hypothesis_has_negation

    
    
    

# Liste de toutes les règles pour paires de textes
TEXT_PAIR_RULES = [
    SymbolicRules.lexical_overlap_rule,
    SymbolicRules.exact_substring_rule,
    SymbolicRules.negation_rule,
    SymbolicRules.excessive_punctuation_rule,
    SymbolicRules.emotion_word_rule,
    SymbolicRules.capital_letter_ratio_rule,
    SymbolicRules.antonym_rule,
    SymbolicRules.numeric_discrepancy_rule
]




""" --------- Fonction de perte symbolique optimisée --------- """

def symbolic_loss(premises, hypotheses, labels, predictions, logits, tokenizer):
    """
    Calcule la pénalité symbolique de manière vectorisée pour optimisation
    
    Args:
        premises: Liste des prémisses (text)
        hypotheses: Liste des hypothèses (text)
        labels: Tensors des labels
        predictions: Tensors des prédictions
        logits: Sortie du modèle
        tokenizer: Tokenizer pour décodage
    """
    # Décodage batch pour optimisation
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Calcul de la confiance
    probs = F.softmax(logits, dim=-1)
    confidences, _ = torch.max(probs, dim=-1)
    
    total_penalty = 0.0
    batch_size = len(premises)
    
    for i in range(batch_size):
        premise = premises[i]
        hypothesis = hypotheses[i]
        pred = decoded_preds[i]
        label = decoded_labels[i]
        confidence = confidences[i].item()
        
        rule_hits = 0
        for rule_fn in ALL_HEURISTIC_RULES:
            try:
                if rule_fn(premise, hypothesis):
                    rule_hits += 1
            except:
                continue
        
        if rule_hits > 0:
            if pred != label:
                total_penalty += 1.0 + 0.1 * rule_hits
            else:
                total_penalty += 0.3 * confidence * (rule_hits / len(ALL_HEURISTIC_RULES))
    
    return total_penalty / batch_size  # Pénalité moyenne






""" --------- Trainer personnalisé avec perte symbolique --------- """

class SymbolicLossTrainer(Seq2SeqTrainer):
    """Trainer personnalisé avec fonction de perte hybride (CrossEntropy + perte symbolique)"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )

        # Perte de type CrossEntropy standard
        ce_loss = outputs.loss

        # Vérification de la présence des champs texte bruts (récupérés pendant preprocessing)
        premises = inputs.get("premise_raw", None)
        hypotheses = inputs.get("hypothesis_raw", None)

        # Si les champs nécessaires à la régularisation symbolique sont présents
        if premises is not None and hypotheses is not None:
            # Prédictions générées (argmax sur les logits)
            predictions = outputs.logits.argmax(dim=-1)

            # Calcul de la régularisation symbolique
            symbol_penalty = symbolic_loss(
                premises=premises,
                hypotheses=hypotheses,
                labels=inputs["labels"],
                predictions=predictions,
                logits=outputs.logits,
                tokenizer=self.tokenizer  # assure-toi que self.tokenizer est bien défini dans l'init
            )

            # Perte totale : combinaison pondérée
            loss = ce_loss + 0.7 * symbol_penalty
        else:
            # Cas normal sans régularisation symbolique
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss







""" --------- Main config --------- """

# Configuration
MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 8
BATCH_SIZE = 16
EPOCHS = 5
USE_KBIT = True
OUTPUT_DIR = "./finetuned-flan-t5-symbolic-V2-5epoch-50%data)"
TRAIN_FILE = "/kaggle/input/multinli-textual-entailment-corpus/train.csv"
VAL_FILE = "/kaggle/input/multinli-textual-entailment-corpus/validation_matched.csv"

# Initialisation
accelerator = Accelerator()
device = accelerator.device

# Chargement du tokenizer et modèle
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

# Préparation pour l'entraînement k-bit
if USE_KBIT:
    base_model = prepare_model_for_kbit_training(base_model)

# Configuration LoRA
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

# Chargement des données
def load_data_to_dataset(file_path):
    df = pd.read_csv(file_path)
    return Dataset.from_pandas(df)

train_dataset1 = load_data_to_dataset(TRAIN_FILE)
val_dataset1 = load_data_to_dataset(VAL_FILE)

train_dataset = train_dataset1.train_test_split(train_size=0.3, seed=42)['train']
val_dataset = val_dataset1.train_test_split(train_size=0.1, seed=42)['train']

# Pré-traitement avec conservation du texte brut
def get_label_name(label):
    return ["entailment", "neutral", "contradiction"][int(label)]

def preprocess_function(examples):
    premises = examples['premise']
    hypotheses = examples['hypothesis']
    labels = [get_label_name(l) for l in examples['label']]
    
    inputs = [f"nli premise: {p} hypothesis: {h}" for p, h in zip(premises, hypotheses)]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=MAX_INPUT_LENGTH, 
        truncation=True, 
        padding="max_length"
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            labels,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length"
        ).input_ids
    
    model_inputs["labels"] = labels
    model_inputs["premise_raw"] = premises
    model_inputs["hypothesis_raw"] = hypotheses
    
    return model_inputs

tokenized_train = train_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=train_dataset.column_names,
    batch_size=512
)

tokenized_val = val_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=val_dataset.column_names,
    batch_size=512
)

# Métriques d'évaluation
def compute_metrics(eval_pred: EvalPrediction):
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
    
    return {"accuracy": acc, "f1_score": f1}

# Configuration d'entraînement
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    logging_strategy="steps",
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=3e-4,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    predict_with_generate=True,
    fp16=True,
    gradient_accumulation_steps=2,
    report_to="none",
    optim="adafactor",
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer, 
    model=model, 
    padding=True,
    pad_to_multiple_of=8
)

# Initialisation du trainer
trainer = SymbolicLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Entraînement
print("Début de l'entraînement avec perte symbolique...")
start_time = time.time()
trainer.train()
end_time = time.time()

# Sauvegarde du modèle
model.save_pretrained(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)

# Évaluation finale
results = trainer.evaluate()
print(f"Résultats d'évaluation: {results}")

# Sauvegarde des métriques
metrics = {
    "training_time": end_time - start_time,
    "eval_accuracy": results["eval_accuracy"],
    "eval_f1_score": results["eval_f1_score"]
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
    
print(f"Entraînement terminé en {metrics['training_time']:.2f} secondes")
print(f"Modèle sauvegardé dans {OUTPUT_DIR}")