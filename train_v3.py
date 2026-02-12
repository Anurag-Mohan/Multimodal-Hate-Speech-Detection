import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import os
import copy


BATCH_SIZE = 32          
LEARNING_RATE = 0.001
EPOCHS = 25              
EMBED_DIM = 100
HIDDEN_DIM = 128
MAX_LEN = 60
VOCAB_SIZE = 6000
IMG_SIZE = 128
PATIENCE = 5             

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Training on: {device}")

nltk.download('punkt', quiet=True)


def build_vocab(texts):
    all_words = []
    for text in texts:
        all_words.extend(word_tokenize(str(text).lower()))
    count = Counter(all_words)
    vocab = {word: i+1 for i, (word, _) in enumerate(count.most_common(VOCAB_SIZE))}
    return vocab

def text_pipeline(text, vocab):
    tokens = word_tokenize(str(text).lower())
    indices = [vocab.get(t, 0) for t in tokens]
    if len(indices) < MAX_LEN:
        indices += [0] * (MAX_LEN - len(indices))
    else:
        indices = indices[:MAX_LEN]
    return torch.tensor(indices, dtype=torch.long)


class MemeDataset(Dataset):
    def __init__(self, jsonl_file, vocab, transform=None):
        self.data = pd.read_json(jsonl_file, lines=True)
        self.vocab = vocab
        self.transform = transform
        
        
        if 'label' not in self.data.columns:
            self.data['label'] = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['img']
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
            
        if self.transform:
            image = self.transform(image)

        text = self.data.iloc[idx]['text']
        text_vec = text_pipeline(text, self.vocab)
        label = int(self.data.iloc[idx]['label'])
        
        return image, text_vec, torch.tensor(label, dtype=torch.float32)



class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.relu = nn.ReLU()
        
        
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) 
        self.flatten = nn.Flatten()
        
        
        self.fc = nn.Linear(256, 128)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        x = self.gap(x)      
        x = self.flatten(x)  
        x = self.relu(self.fc(x))
        return x

class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(TextBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 128) 

    def forward(self, x):
        x = self.embedding(x)
        self.lstm.flatten_parameters() 
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden_cat)

class MultimodalNet(nn.Module):
    def __init__(self, vocab_size):
        super(MultimodalNet, self).__init__()
        self.cnn = CustomCNN()
        self.lstm = TextBiLSTM(vocab_size, EMBED_DIM, HIDDEN_DIM)
        
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Dropout(0.5),     
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text):
        visual_feat = self.cnn(image)
        text_feat = self.lstm(text)
        combined = torch.cat((visual_feat, text_feat), dim=1)
        output = self.fusion(combined)
        return output


def train_model():
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("--- Preparing Data ---")
    df_train = pd.read_json('train.jsonl', lines=True)
    vocab = build_vocab(df_train['text'].tolist())
    
    train_dataset = MemeDataset('train.jsonl', vocab, train_transform)
    
    val_dataset = MemeDataset('dev.jsonl', vocab, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MultimodalNet(len(vocab)).to(device)
    criterion = nn.BCELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    print(f"--- Starting V3 Training (GAP + BatchNorm + Early Stopping) ---")
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(EPOCHS):
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for imgs, texts, labels in train_loader:
            imgs, texts, labels = imgs.to(device), texts.to(device), labels.to(device)
            labels = labels.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(imgs, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, texts, labels in val_loader:
                imgs, texts, labels = imgs.to(device), texts.to(device), labels.to(device)
                labels = labels.unsqueeze(1)
                
                outputs = model(imgs, texts)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_train_acc = 100 * train_correct / train_total
        epoch_val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {epoch_val_acc:.2f}% | LR: {current_lr}")
        
        scheduler.step(avg_val_loss)

        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0 
            print(f"  >>> New Best Validation Accuracy! Saving model state...")
        else:
            patience_counter += 1
            print(f"  >>> No improvement for {patience_counter} epochs.")
            
        if patience_counter >= PATIENCE:
            print("🛑 Early Stopping Triggered! Restoring best model...")
            break

    
    torch.save(best_model_wts, 'meme_model_v3.pth')
    import pickle
    with open('vocab_v3.pkl', 'wb') as f:
        pickle.dump(vocab, f)
        
    print(f"\n✅ Training Complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    print("Saved as 'meme_model_v3.pth' and 'vocab_v3.pkl'")

if __name__ == "__main__":
    train_model()