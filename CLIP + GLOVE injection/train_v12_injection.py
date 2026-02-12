import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE
import requests
import zipfile
import os


BATCH_SIZE = 32
IMG_SIZE = 224          
EMBED_DIM = 300         
HIDDEN_DIM = 256
VOCAB_SIZE = 15000      
EPOCHS_PHASE_1 = 12     
EPOCHS_PHASE_2 = 15     

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 V12: 'Trojan Horse' Architecture (Weight Injection) on: {device}")

nltk.download('punkt', quiet=True)


def download_glove_300d():
    if not os.path.exists('glove.6B.300d.txt'):
        print("--- Downloading 300d GloVe (High-Res Text Memory)... ---")
        url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
        r = requests.get(url)
        with open("glove.6B.zip", "wb") as f: f.write(r.content)
        print("Unzipping...")
        with zipfile.ZipFile("glove.6B.zip", "r") as zip_ref: zip_ref.extractall(".")
        print("✅ High-Res GloVe Ready.")

download_glove_300d()



class CustomVisualNet(nn.Module):
    def __init__(self):
        super(CustomVisualNet, self).__init__()
        print("--- Initializing Custom Visual Cortex ---")
        print("💉 INJECTING MEMORY: Loading ImageNet Synaptic Priors...")
        
        
        
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        
        
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        
        self.adapter = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, x):
        
        features = self.feature_extractor(x)
        
        return self.adapter(features)

class TextBiLSTM_Pro(nn.Module):
    def __init__(self, vocab_size, embedding_matrix):
        super(TextBiLSTM_Pro, self).__init__()
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, num_layers=2, 
                          batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(HIDDEN_DIM * 2, 512) 

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(hidden_cat)

class MultimodalNet(nn.Module):
    def __init__(self, vocab_size, embedding_matrix):
        super(MultimodalNet, self).__init__()
        self.cnn = CustomVisualNet() 
        self.lstm = TextBiLSTM_Pro(vocab_size, embedding_matrix)
        
        
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text):
        img_feat = self.cnn(image)
        text_feat = self.lstm(text)
        combined = torch.cat((img_feat, text_feat), dim=1)
        return self.fusion(combined)
    
    def extract_features(self, image, text):
        img_feat = self.cnn(image)
        text_feat = self.lstm(text)
        return torch.cat((img_feat, text_feat), dim=1)


def load_glove_vectors(vocab):
    embeddings_index = {}
    with open('glove.6B.300d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    matrix = np.zeros((len(vocab) + 1, EMBED_DIM))
    for word, i in vocab.items():
        v = embeddings_index.get(word)
        if v is not None: matrix[i] = v
    return torch.tensor(matrix, dtype=torch.float32)

def build_vocab(texts):
    all_words = [w for t in texts for w in word_tokenize(str(t).lower())]
    return {w: i+1 for i, (w, _) in enumerate(Counter(all_words).most_common(VOCAB_SIZE))}

def text_pipeline(text, vocab):
    idxs = [vocab.get(t, 0) for t in word_tokenize(str(text).lower())]
    return torch.tensor((idxs + [0]*60)[:60], dtype=torch.long)

class MemeDataset(Dataset):
    def __init__(self, jsonl_file, vocab, transform=None):
        self.data = pd.read_json(jsonl_file, lines=True)
        self.vocab = vocab
        self.transform = transform
        if 'label' not in self.data.columns: self.data['label'] = 0
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try: img = Image.open(row['img']).convert("RGB")
        except: img = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        if self.transform: img = self.transform(img)
        return img, text_pipeline(row['text'], self.vocab), torch.tensor(row['label'], dtype=torch.float32)


def run_trojan_training():
    
    df_train = pd.read_json('train.jsonl', lines=True)
    df_dev = pd.read_json('dev.jsonl', lines=True)
    vocab = build_vocab(df_train['text'].tolist() + df_dev['text'].tolist())
    emb_matrix = load_glove_vectors(vocab)
    
    
    train_tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = MemeDataset('train.jsonl', vocab, train_tfm)
    val_ds = MemeDataset('dev.jsonl', vocab, val_tfm)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = MultimodalNet(len(vocab), emb_matrix).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, 
                                            steps_per_epoch=len(train_loader), epochs=EPOCHS_PHASE_1)

    print("--- PHASE 1: Training Adapter Layers (Feature Alignment) ---")
    for epoch in range(EPOCHS_PHASE_1):
        model.train()
        losses = []
        for img, txt, lbl in train_loader:
            img, txt, lbl = img.to(device), txt.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img, txt), lbl.unsqueeze(1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1} | Loss: {np.mean(losses):.4f}")

    print("\n--- PHASE 2: SMOTE Injection (Balancing the Knowledge) ---")
    model.eval()
    feats_list, lbls_list = [], []
    with torch.no_grad():
        for img, txt, lbl in train_loader:
            img, txt = img.to(device), txt.to(device)
            feats_list.append(model.extract_features(img, txt).cpu().numpy())
            lbls_list.append(lbl.numpy())
            
    X = np.vstack(feats_list)
    y = np.hstack(lbls_list)
    
    print(f"Applying SMOTE to {X.shape} features...")
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    
    smote_ds = TensorDataset(torch.FloatTensor(X_res), torch.FloatTensor(y_res))
    smote_loader = DataLoader(smote_ds, batch_size=64, shuffle=True)
    
    
    head_opt = optim.Adam(model.fusion.parameters(), lr=0.0005)
    best_acc = 0.0
    
    print("--- PHASE 3: Final Classification Training ---")
    for epoch in range(EPOCHS_PHASE_2):
        model.fusion.train()
        for f, l in smote_loader:
            f, l = f.to(device), l.to(device)
            head_opt.zero_grad()
            loss = criterion(model.fusion(f), l.unsqueeze(1))
            loss.backward()
            head_opt.step()
            
        
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for img, txt, lbl in val_loader:
                img, txt, lbl = img.to(device), txt.to(device), lbl.to(device)
                pred = (model(img, txt) > 0.5).float()
                correct += (pred == lbl.unsqueeze(1)).sum().item()
                total += lbl.size(0)
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1} | Val Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'meme_model_v12_injected.pth')
            import pickle
            with open('vocab_v12.pkl', 'wb') as f: pickle.dump(vocab, f)

    print(f"\n✅ Final Accuracy with Injected Memory: {best_acc:.2f}%")

if __name__ == "__main__":
    run_trojan_training()