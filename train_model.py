import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
import os
import json

# Load your dataset from a CSV file
df = pd.read_csv('./data.csv')  # Update the path to your dataset

# Encode labels
label_mapping = {label: idx for idx, label in enumerate(df['Label'].unique())}
df['Label'] = df['Label'].map(label_mapping)

# Save the label mapping to a file
with open('label_mapping.json', 'w') as f:
    json.dump(label_mapping, f)

# Split the data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text
train_encodings = tokenizer(train_df['Query'].tolist(), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_df['Query'].tolist(), truncation=True, padding=True, max_length=128)

train_labels = train_df['Label'].tolist()
val_labels = val_df['Label'].tolist()

class ExpenseReportDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = ExpenseReportDataset(train_encodings, train_labels)
val_dataset = ExpenseReportDataset(val_encodings, val_labels)

# Directory to save/load the model
model_dir = './expense_report_model'

if not os.path.exists(model_dir):
    # Train the model if it doesn't exist
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_mapping))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optim = AdamW(model.parameters(), lr=5e-5)

    # Training loop with progress bars and intermediate logging
    for epoch in range(3):  # number of epochs
        model.train()
        train_loss = 0
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch + 1}: average training loss = {avg_train_loss}')

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / len(val_dataset)
        print(f'Epoch {epoch + 1}: validation loss = {val_loss}, accuracy = {accuracy}')

    # Save the model and tokenizer
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
else:
    print(f"Model already exists at {model_dir}. Training skipped.")
