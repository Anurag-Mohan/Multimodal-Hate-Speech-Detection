import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import os
import requests
import zipfile


BATCH_SIZE = 32
IMG_SIZE = 224 
EMBED_DIM = 100
HIDDEN_DIM = 128
VOCAB_SIZE = 10000
EPOCHS = 20
DISTILLATION_WEIGHT = 2.0 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 V11: Knowledge Distillation (Teacher-Student) on: {device}")

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
    if len(indices) < 60: indices += [0] * (60 - len(indices))
    else: indices = indices[:60]
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
        if self.transform: image = self.transform(image)
        text = self.data.iloc[idx]['text']
        text_vec = text_pipeline(text, self.vocab)
        label = int(self.data.iloc[idx]['label'])
        return image, text_vec, torch.tensor(label, dtype=torch.float32)



class TeacherResNet(nn.Module):
    def __init__(self):
        super(TeacherResNet, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        self.features = nn.Sequential(*list(resnet.children())[:-1]) 
    
    def forward(self, x):
        with torch.no_grad(): 
            out = self.features(x)
            return out.flatten(1) 


class StudentCNN(nn.Module):
    def __init__(self):
        super(StudentCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        
        
        
        
        self.mimic_layer = nn.Linear(256, 512) 

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.gap(x)
        my_features = self.flatten(x) 
        
        
        mimic_features = self.mimic_layer(my_features) 
        
        return my_features, mimic_features

class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size):
        super(TextBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(HIDDEN_DIM * 2, 256) 

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden_cat)

class MultimodalStudent(nn.Module):
    def __init__(self, vocab_size):
        super(MultimodalStudent, self).__init__()
        self.cnn = StudentCNN()
        self.lstm = TextBiLSTM(vocab_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text):
        img_feat, mimic_feat = self.cnn(image)
        text_feat = self.lstm(text)
        
        
        combined = torch.cat((img_feat, text_feat), dim=1)
        output = self.classifier(combined)
        
        return output, mimic_feat


def train_pipeline():
    
    df_train = pd.read_json('train.jsonl', lines=True)
    df_dev = pd.read_json('dev.jsonl', lines=True)
    full_text = df_train['text'].tolist() + df_dev['text'].tolist()
    vocab = build_vocab(full_text)
    
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = MemeDataset('train.jsonl', vocab, transform)
    val_dataset = MemeDataset('dev.jsonl', vocab, transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    
    teacher_model = TeacherResNet().to(device)
    teacher_model.eval() 
    
    student_model = MultimodalStudent(len(vocab)).to(device)
    
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    
    criterion_cls = nn.BCELoss() 
    criterion_mimic = nn.MSELoss() 

    print("--- Starting Distillation Training ---")
    print(f"Teacher: ResNet-18 (Frozen) | Student: CustomCNN")
    
    best_acc = 0.0

    for epoch in range(EPOCHS):
        student_model.train()
        total_loss = 0
        
        for img, txt, lbl in train_loader:
            img, txt, lbl = img.to(device), txt.to(device), lbl.to(device)
            
            optimizer.zero_grad()
            
            
            with torch.no_grad():
                teacher_features = teacher_model(img) 
            
            
            student_pred, student_mimic_features = student_model(img, txt)
            
            
            loss_classification = criterion_cls(student_pred, lbl.unsqueeze(1))
            loss_mimic = criterion_mimic(student_mimic_features, teacher_features)
            
            
            loss = loss_classification + (DISTILLATION_WEIGHT * loss_mimic)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        
        student_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for img, txt, lbl in val_loader:
                img, txt, lbl = img.to(device), txt.to(device), lbl.to(device)
                pred, _ = student_model(img, txt) 
                pred = (pred > 0.5).float()
                correct += (pred == lbl.unsqueeze(1)).sum().item()
                total += lbl.size(0)
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Dev Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            
            torch.save(student_model.state_dict(), 'meme_model_v11_distill.pth')
            import pickle
            with open('vocab_v11.pkl', 'wb') as f: pickle.dump(vocab, f)

    print(f"\n✅ Final Distilled Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_pipeline()