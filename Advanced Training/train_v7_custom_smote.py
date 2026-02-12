import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE
import copy


BATCH_SIZE = 32
IMG_SIZE = 128
EMBED_DIM = 100
HIDDEN_DIM = 128
VOCAB_SIZE = 7000
EPOCHS_PHASE_1 = 15  
EPOCHS_PHASE_2 = 10  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 V8: Custom CNN+BiLSTM with SMOTE Innovation on: {device}")

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
    if len(indices) < 60:
        indices += [0] * (60 - len(indices))
    else:
        indices = indices[:60]
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



class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.gap(x)
        x = self.flatten(x) 
        return x

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

class MultimodalNet(nn.Module):
    def __init__(self, vocab_size):
        super(MultimodalNet, self).__init__()
        self.cnn = CustomCNN()
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
        img_feat = self.cnn(image)
        text_feat = self.lstm(text)
        combined = torch.cat((img_feat, text_feat), dim=1) 
        return self.classifier(combined)
    
    
    def extract_features(self, image, text):
        img_feat = self.cnn(image)
        text_feat = self.lstm(text)
        return torch.cat((img_feat, text_feat), dim=1)


def train_pipeline():
    
    df_train = pd.read_json('train.jsonl', lines=True)
    df_dev = pd.read_json('dev.jsonl', lines=True)
    full_text = df_train['text'].tolist() + df_dev['text'].tolist()
    vocab = build_vocab(full_text)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = MemeDataset('train.jsonl', vocab, transform)
    val_dataset = MemeDataset('dev.jsonl', vocab, transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MultimodalNet(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print("--- PHASE 1: Training Custom Extractors ---")
    for epoch in range(EPOCHS_PHASE_1):
        model.train()
        total_loss = 0
        for img, txt, lbl in train_loader:
            img, txt, lbl = img.to(device), txt.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(img, txt)
            loss = criterion(out, lbl.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS_PHASE_1} | Loss: {total_loss/len(train_loader):.4f}")

    print("\n--- PHASE 2: Applying SMOTE Innovation (Tchokote et al.) ---")
    
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for img, txt, lbl in train_loader:
            img, txt = img.to(device), txt.to(device)
            feats = model.extract_features(img, txt) 
            all_features.append(feats.cpu().numpy())
            all_labels.append(lbl.numpy())
    
    X_train = np.vstack(all_features)
    y_train = np.hstack(all_labels)
    
    
    print(f"Original Shape: {X_train.shape} | Balancing classes...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"New SMOTE Shape: {X_res.shape}")
    
    
    smote_dataset = TensorDataset(torch.FloatTensor(X_res), torch.FloatTensor(y_res))
    smote_loader = DataLoader(smote_dataset, batch_size=64, shuffle=True)
    
    
    
    for param in model.cnn.parameters(): param.requires_grad = False
    for param in model.lstm.parameters(): param.requires_grad = False
    
    optimizer_head = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    print("--- Retraining Classifier on SMOTE Features ---")
    best_acc = 0.0
    for epoch in range(EPOCHS_PHASE_2):
        model.classifier.train()
        for feats, lbl in smote_loader:
            feats, lbl = feats.to(device), lbl.to(device)
            optimizer_head.zero_grad()
            out = model.classifier(feats)
            loss = criterion(out, lbl.unsqueeze(1))
            loss.backward()
            optimizer_head.step()
            
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for img, txt, lbl in val_loader:
                img, txt, lbl = img.to(device), txt.to(device), lbl.to(device)
                out = model(img, txt)
                pred = (out > 0.5).float()
                correct += (pred == lbl.unsqueeze(1)).sum().item()
                total += lbl.size(0)
        
        acc = 100 * correct / total
        print(f"Phase 2 Epoch {epoch+1} | Validation Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'meme_model_v8_smote.pth')
            import pickle
            with open('vocab_v8.pkl', 'wb') as f: pickle.dump(vocab, f)

    print(f"\n✅ Final Accuracy with SMOTE Innovation: {best_acc:.2f}%")

if __name__ == "__main__":
    train_pipeline()