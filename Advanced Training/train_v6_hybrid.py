import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
EPOCHS = 25              
MAX_LEN = 60
EMBED_DIM = 100
HIDDEN_DIM = 128
VOCAB_SIZE = 7000
IMG_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Training V6 Hybrid (V4 CNN + V5 Co-Attention) on: {device}")

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
        if 'label' not in self.data.columns: self.data['label'] = 0

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
        
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.bn4(self.conv4(x)))
        
        return x


class CoAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CoAttention, self).__init__()
        self.W_b = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_v = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_q = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.w_hv = nn.Parameter(torch.randn(hidden_dim, 1))
        self.w_hq = nn.Parameter(torch.randn(hidden_dim, 1))

    def forward(self, V, Q):
        
        
        
        
        C = torch.matmul(Q, torch.matmul(self.W_b, V.permute(0, 2, 1)))
        C = torch.tanh(C)
        
        
        H_v = torch.tanh(torch.matmul(self.W_v, V.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))
        a_v = F.softmax(torch.matmul(self.w_hv.t(), H_v), dim=2)
        
        H_q = torch.tanh(torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V.permute(0, 2, 1)), C.permute(0, 2, 1)))
        a_q = F.softmax(torch.matmul(self.w_hq.t(), H_q), dim=2)
        
        v_hat = torch.matmul(a_v, V).squeeze(1)
        q_hat = torch.matmul(a_q, Q).squeeze(1)
        
        return v_hat, q_hat

class MultimodalNet(nn.Module):
    def __init__(self, vocab_size):
        super(MultimodalNet, self).__init__()
        self.cnn = CustomCNN()
        
        self.embedding = nn.Embedding(vocab_size + 1, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.text_proj = nn.Linear(HIDDEN_DIM * 2, 128)
        
        self.co_attn = CoAttention(128)
        
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text):
        
        img_feat = self.cnn(image) 
        b, c, h, w = img_feat.size()
        img_feat = img_feat.view(b, c, h*w).permute(0, 2, 1) 
        
        
        x = self.embedding(text)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        text_feat = self.text_proj(lstm_out)
        
        
        attended_img, attended_text = self.co_attn(img_feat, text_feat)
        
        combined = torch.cat((attended_img, attended_text), dim=1)
        output = self.fusion(combined)
        return output


def train_model():
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
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
    df_dev = pd.read_json('dev.jsonl', lines=True)
    full_text = df_train['text'].tolist() + df_dev['text'].tolist()
    vocab = build_vocab(full_text)
    
    train_dataset = MemeDataset('train.jsonl', vocab, train_transform)
    val_dataset = MemeDataset('dev.jsonl', vocab, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MultimodalNet(len(vocab)).to(device)
    criterion = nn.BCELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, steps_per_epoch=len(train_loader), epochs=EPOCHS)

    print(f"--- Starting V6 Hybrid Training ---")
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

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
            scheduler.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, texts, labels in val_loader:
                imgs, texts, labels = imgs.to(device), texts.to(device), labels.to(device)
                labels = labels.unsqueeze(1)
                
                outputs = model(imgs, texts)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_train_acc = 100 * train_correct / train_total
        epoch_val_acc = 100 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {epoch_train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}% | LR: {current_lr:.6f}")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  >>> New Best! Saving...")

    torch.save(best_model_wts, 'meme_model_v6.pth')
    import pickle
    with open('vocab_v6.pkl', 'wb') as f:
        pickle.dump(vocab, f)
        
    print(f"\n✅ V6 Complete. Best Validation: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model()