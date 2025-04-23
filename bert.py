# BERT Text Classification: Three Models with Custom Learning Rates and Freezing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    IntervalStrategy
)
from sklearn.metrics import accuracy_score, classification_report

# Load Data
train_df = pd.read_csv('train.csv')
train_df['Score'] = train_df['Score'] - 1
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['Text'], train_df['Score'], test_size=0.05, random_state=42, stratify=train_df['Score']
)
train_data = pd.DataFrame({'Text': train_texts, 'Score': train_labels})
val_data = pd.DataFrame({'Text': val_texts, 'Score': val_labels})

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_function(examples):
    return tokenizer(examples['Text'], truncation=True, padding='max_length', max_length=128)

train_dataset = Dataset.from_pandas(train_data).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_data).map(tokenize_function, batched=True)
test_df = pd.read_csv("test.csv")
test_df["Score"] = test_df["Score"] - 1
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("Score", "labels")
val_dataset = val_dataset.rename_column("Score", "labels")
test_dataset = test_dataset.rename_column("Score", "labels")

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Metrics
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {'accuracy': accuracy_score(p.label_ids, preds)}

# Training Args
training_args_base = {
    'eval_strategy': IntervalStrategy.EPOCH,
    'save_strategy': IntervalStrategy.EPOCH,
    'logging_strategy': IntervalStrategy.EPOCH,
    'per_device_train_batch_size': 256,
    'per_device_eval_batch_size': 2048,
    'num_train_epochs': 20,
    'weight_decay': 3e-2,
    'learning_rate': 3e-4,
    'logging_dir': './logs',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'accuracy',
    'report_to': 'none',
    'lr_scheduler_type': 'constant',
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
}

# Helper: Grouped Optimizer

def get_optimizer_grouped_parameters(model, lr, reduced_lr_layers=[], reduce_factor=0.1):
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        weight_decay = 0.0 if any(nd in name for nd in no_decay) else 1e-2
        apply_reduced_lr = any(layer_name in name for layer_name in reduced_lr_layers)
        grouped_parameters.append({
            "params": [param],
            "weight_decay": weight_decay,
            "lr": lr * reduce_factor if apply_reduced_lr else lr
        })
    return grouped_parameters

plot_logs = {}

# ===================
# MODEL 1: Full model
# ===================
model1 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5,hidden_dropout_prob=0.1,attention_probs_dropout_prob=0.1)
for param in model1.bert.parameters():
    param.requires_grad = False
for param in model1.classifier.parameters():
    param.requires_grad = True
    
for i in range(12):
    for param in model1.bert.encoder.layer[i].parameters():
        param.requires_grad = True

     
opt_grouped_params1 = get_optimizer_grouped_parameters(
    model1,
    lr=training_args_base['learning_rate'],
    reduced_lr_layers=['encoder'],
    reduce_factor=0.1
)

args1 = TrainingArguments(**training_args_base, output_dir="./results/model1")
trainer1 = Trainer(
    model=model1,
    args=args1,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    optimizers=(torch.optim.AdamW(opt_grouped_params1), None)
)
trainer1.train()

preds_val_1 = trainer1.predict(val_dataset)
preds_test_1 = trainer1.predict(test_dataset)
pred_labels_test_1 = np.argmax(preds_test_1.predictions, axis=1)
print("\nValidation Accuracy (Model 1):", accuracy_score(preds_val_1.label_ids, np.argmax(preds_val_1.predictions, axis=1)))
print("Test Accuracy (Model 1):", accuracy_score(preds_test_1.label_ids, pred_labels_test_1))
print("Classification Report (Model 1):\n", classification_report(preds_test_1.label_ids, pred_labels_test_1, digits=3))
plot_logs['model1'] = trainer1.state.log_history

# ===================
# MODEL 2: Unfreeze Last two
# ===================
model2 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5,hidden_dropout_prob=0.1,attention_probs_dropout_prob=0.1)
for param in model2.bert.parameters():
    param.requires_grad = False
for param in model2.classifier.parameters():
    param.requires_grad = True
for param in model2.bert.encoder.layer[11].parameters():
    param.requires_grad = True
for param in model2.bert.encoder.layer[10].parameters():
    param.requires_grad = True
    
args2 = TrainingArguments(**training_args_base, output_dir="./results/model2")

opt_grouped_params2 = get_optimizer_grouped_parameters(
    model2,
    lr=training_args_base['learning_rate'],
    reduced_lr_layers=['encoder'],
    reduce_factor=0.01
)

trainer2 = Trainer(
    model=model2,
    args=args2,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    optimizers=(torch.optim.AdamW(opt_grouped_params2), None)
)
trainer2.train()

preds_val_2 = trainer2.predict(val_dataset)
preds_test_2 = trainer2.predict(test_dataset)
pred_labels_test_2 = np.argmax(preds_test_2.predictions, axis=1)
print("\nValidation Accuracy (Model 2):", accuracy_score(preds_val_2.label_ids, np.argmax(preds_val_2.predictions, axis=1)))
print("Test Accuracy (Model 2):", accuracy_score(preds_test_2.label_ids, pred_labels_test_2))
print("Classification Report (Model 2):\n", classification_report(preds_test_2.label_ids, pred_labels_test_2, digits=3))
plot_logs['model2'] = trainer2.state.log_history

# ===================
# MODEL 3: Freeze All Layer
# ===================
model3 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
for param in model3.bert.parameters():
    param.requires_grad = False
    
for param in model3.classifier.parameters():
    param.requires_grad = True
args3 = TrainingArguments(**training_args_base, output_dir="./results/model3")
trainer3 = Trainer(
    model=model3,
    args=args3,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)
trainer3.train()

preds_val_3 = trainer3.predict(val_dataset)
preds_test_3 = trainer3.predict(test_dataset)
pred_labels_test_3 = np.argmax(preds_test_3.predictions, axis=1)
print("\nValidation Accuracy (Model 3):", accuracy_score(preds_val_3.label_ids, np.argmax(preds_val_3.predictions, axis=1)))
print("Test Accuracy (Model 3):", accuracy_score(preds_test_3.label_ids, pred_labels_test_3))
print("Classification Report (Model 3):\n", classification_report(preds_test_3.label_ids, pred_labels_test_3, digits=3))
plot_logs['model3'] = trainer3.state.log_history

# ===================
# PLOTS
# ===================
os.makedirs("plots", exist_ok=True)

def extract_logs(logs):
    train_loss = [log['loss'] for log in logs if 'loss' in log]
    eval_loss = [log['eval_loss'] for log in logs if 'eval_loss' in log]
    eval_acc = [log['eval_accuracy'] for log in logs if 'eval_accuracy' in log]
    epochs = list(range(1, len(eval_loss) + 1))
    return train_loss, eval_loss, eval_acc, epochs

train_loss_1, eval_loss_1, eval_acc_1, epochs = extract_logs(plot_logs['model1'])
train_loss_2, eval_loss_2, eval_acc_2, _ = extract_logs(plot_logs['model2'])
train_loss_3, eval_loss_3, eval_acc_3, _ = extract_logs(plot_logs['model3'])

plt.figure()
plt.plot(range(1, len(train_loss_1)+1), train_loss_1, label='Tune-full Train')
plt.plot(epochs, eval_loss_1, label='Tune-full Val')
plt.plot(range(1, len(train_loss_2)+1), train_loss_2, label='Tune-last2 Train')
plt.plot(epochs, eval_loss_2, label='Tune-last2 Val')
plt.plot(range(1, len(train_loss_3)+1), train_loss_3, label='Tune-linear Train')
plt.plot(epochs, eval_loss_3, label='Tune-linear Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.savefig('plots/all_models_loss.png')
plt.close()

plt.figure()
plt.plot(epochs, eval_acc_1, label='Tune-full')
plt.plot(epochs, eval_acc_2, label='Tune-last2')
plt.plot(epochs, eval_acc_3, label='Tune-linear')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.savefig('plots/all_models_accuracy.png')
plt.close()