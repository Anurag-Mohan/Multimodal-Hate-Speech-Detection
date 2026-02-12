import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image, ImageFile
import pandas as pd
import numpy as np
import os
import json
import collections
import nltk
import random
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast  


ImageFile.LOAD_TRUNCATED_IMAGES = True




SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
ARCHIVE_5_ROOT = os.path.dirname(SCRIPT_DIR)            
ARCHIVE_6_ROOT = os.path.abspath(os.path.join(ARCHIVE_5_ROOT, "../archive (6)")) 

def find_file(filename, search_paths):
    for path in search_paths:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path): return full_path
    return None

FB_TRAIN = find_file("train.jsonl", [SCRIPT_DIR, ARCHIVE_5_ROOT])
FB_DEV = find_file("dev.jsonl", [SCRIPT_DIR, ARCHIVE_5_ROOT])
FB_IMG_DIR = find_file("img", [SCRIPT_DIR, ARCHIVE_5_ROOT])
MMHS_GT = find_file("mmhs150k-dataset/MMHS150K_GT.json", [ARCHIVE_6_ROOT])
MMHS_IMG_DIR = find_file("mmhs150k-dataset/img_resized", [ARCHIVE_6_ROOT])
GLOVE_PATH = find_file("glove.6B.300d.txt", [SCRIPT_DIR, ARCHIVE_5_ROOT])

CONFIG = {
    'BATCH_SIZE': 32,       
    'LR': 5e-5,      
    'EPOCHS_STAGE1': 1,     
    'EPOCHS_STAGE2': 8,     
    'IMG_SIZE': 224,
    'MMHS_LIMIT': 30000,    
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'SAVE_PATH': os.path.join(SCRIPT_DIR, 'model_v16_turbo.pth'),
    'WORKERS': 2            
}

print(f"🚀 TURBO SYSTEM ONLINE: Running on {CONFIG['DEVICE']}")
if str(CONFIG['DEVICE']) == 'cuda':
    print(f"⚡ AMP Enabled: Training will be ~2x faster.")

nltk.download('punkt', quiet=True)




class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
    def __len__(self): return len(self.itos)
    def build_vocabulary(self, sentence_list):
        frequencies = collections.Counter()
        idx = 4
        for sentence in sentence_list:
            for word in word_tokenize(str(sentence).lower()):
                frequencies[word] += 1
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx; self.itos[idx] = word; idx += 1
    def numericalize(self, text):
        return [self.stoi.get(t, 3) for t in word_tokenize(str(text).lower())]

def load_glove_embeddings(vocab, glove_path):
    if not glove_path: return None
    print("🔌 Loading GloVe Embeddings...")
    embeddings_index = {}
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
    except: return None
    matrix = np.zeros((len(vocab), 300))
    for word, i in vocab.stoi.items():
        v = embeddings_index.get(word)
        if v is not None: matrix[i] = v
    return torch.tensor(matrix, dtype=torch.float32)




class MMHSDataset(Dataset):
    def __init__(self, json_path, img_dir, vocab, transform=None, limit=None):
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        self.data = []
        if json_path and os.path.exists(json_path):
            print("📂 Parsing MMHS150K Data...")
            with open(json_path, 'r') as f:
                raw_data = json.load(f)
            all_items = list(raw_data.items())
            if limit:
                print(f"   ⚡ Speed Hack: Randomly selecting {limit} samples...")
                random.shuffle(all_items)
                all_items = all_items[:limit]
            for k, v in all_items:
                labels = v.get('labels', [])
                if not labels: continue
                label = 1 if sum(labels) >= 2 else 0
                img_name = f"{k}.jpg"
                text = v.get('tweet_text', "")
                if os.path.exists(os.path.join(img_dir, img_name)):
                    self.data.append((img_name, text, label))
            print(f"   ✅ Loaded {len(self.data)} MMHS samples.")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img_name, text, label = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try: image = Image.open(img_path).convert("RGB")
        except: image = Image.new('RGB', (224, 224))
        if self.transform: image = self.transform(image)
        tokens = self.vocab.numericalize(text)
        if len(tokens) < 60: tokens += [0] * (60 - len(tokens))
        else: tokens = tokens[:60]
        return image, torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

class FacebookDataset(Dataset):
    def __init__(self, json_path, img_dir, vocab, transform=None):
        self.df = pd.read_json(json_path, lines=True)
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['img'])
        try: image = Image.open(img_path).convert("RGB")
        except: image = Image.new('RGB', (224, 224))
        if self.transform: image = self.transform(image)
        tokens = self.vocab.numericalize(row['text'])
        if len(tokens) < 60: tokens += [0] * (60 - len(tokens))
        else: tokens = tokens[:60]
        return image, torch.tensor(tokens, dtype=torch.long), torch.tensor(row['label'], dtype=torch.float32)




class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        full_layers = list(resnet.children())[:-1] 
        self.backbone = nn.Sequential(*full_layers)
        
        
        
        count = 0
        for param in self.backbone.parameters():
            if count < 100: param.requires_grad = False
            else: param.requires_grad = True
            count += 1
            
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU())
    def forward(self, x): return self.fc(self.backbone(x))

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_weights):
        super().__init__()
        if embed_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(embed_weights, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, 300, padding_idx=0)
        self.lstm = nn.LSTM(300, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        _, (h, _) = self.lstm(self.embedding(x))
        return self.dropout(self.fc(torch.cat((h[-2], h[-1]), dim=1)))

class TrojanModel(nn.Module):
    def __init__(self, vocab_size, embed_weights):
        super().__init__()
        self.vis = VisualEncoder()
        self.txt = TextEncoder(vocab_size, embed_weights)
        self.head = nn.Sequential(
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 1) 
        )
    def forward(self, img, txt):
        return self.head(torch.cat((self.vis(img), self.txt(txt)), dim=1))




def run_pipeline():
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("🚧 Building Unified Vocabulary...")
    df_fb = pd.read_json(FB_TRAIN, lines=True)
    vocab = Vocabulary()
    vocab.build_vocabulary(df_fb['text'].tolist())
    
    glove = load_glove_embeddings(vocab, GLOVE_PATH)
    model = TrojanModel(len(vocab), glove).to(CONFIG['DEVICE'])
    
    
    criterion = nn.BCEWithLogitsLoss() 
    scaler = GradScaler() 
    
    
    if MMHS_GT:
        print("\n" + "="*40 + "\nSTAGE 1: PRE-TRAINING (30k Samples)\n" + "="*40)
        mmhs_dataset = MMHSDataset(MMHS_GT, MMHS_IMG_DIR, vocab, transform, limit=CONFIG['MMHS_LIMIT'])
        mmhs_loader = DataLoader(mmhs_dataset, batch_size=CONFIG['BATCH_SIZE'], 
                               shuffle=True, num_workers=CONFIG['WORKERS'], pin_memory=True)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        for epoch in range(CONFIG['EPOCHS_STAGE1']):
            model.train()
            loop = tqdm(mmhs_loader, desc=f"Pre-training")
            for img, txt, lbl in loop:
                img, txt, lbl = img.to(CONFIG['DEVICE']), txt.to(CONFIG['DEVICE']), lbl.to(CONFIG['DEVICE'])
                optimizer.zero_grad()
                
                
                with autocast():
                    output = model(img, txt).squeeze()
                    loss = criterion(output, lbl)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                loop.set_postfix(loss=loss.item())
                
    
    print("\n" + "="*40 + "\nSTAGE 2: FINE-TUNING (FACEBOOK)\n" + "="*40)
    fb_train = FacebookDataset(FB_TRAIN, FB_IMG_DIR, vocab, transform)
    fb_dev = FacebookDataset(FB_DEV, FB_IMG_DIR, vocab, transform)
    
    targets = fb_train.df['label'].values
    class_weights = 1. / np.bincount(targets.astype(int))
    sample_weights = [class_weights[int(t)] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(fb_train, batch_size=CONFIG['BATCH_SIZE'], sampler=sampler, 
                            num_workers=CONFIG['WORKERS'], pin_memory=True)
    val_loader = DataLoader(fb_dev, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, 
                          num_workers=CONFIG['WORKERS'], pin_memory=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
    
    best_acc = 0.0
    
    for epoch in range(CONFIG['EPOCHS_STAGE2']):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for img, txt, lbl in loop:
            img, txt, lbl = img.to(CONFIG['DEVICE']), txt.to(CONFIG['DEVICE']), lbl.to(CONFIG['DEVICE'])
            optimizer.zero_grad()
            
            with autocast():
                output = model(img, txt).squeeze()
                loss = criterion(output, lbl)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        
        model.eval()
        correct = 0; total = 0
        val_loss = 0
        with torch.no_grad():
            for img, txt, lbl in val_loader:
                img, txt, lbl = img.to(CONFIG['DEVICE']), txt.to(CONFIG['DEVICE']), lbl.to(CONFIG['DEVICE'])
                with autocast():
                    out = model(img, txt).squeeze()
                    val_loss += criterion(out, lbl).item()
                
                
                probs = torch.sigmoid(out)
                preds = (probs > 0.5).float()
                correct += (preds == lbl).sum().item()
                total += lbl.size(0)
        
        acc = 100 * correct / total
        avg_val = val_loss/len(val_loader)
        scheduler.step(avg_val)
        
        print(f"   Val Acc: {acc:.2f}% | Val Loss: {avg_val:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), CONFIG['SAVE_PATH'])
            print(f"   💾 Best Model Saved: {acc:.2f}%")

if __name__ == "__main__":
    run_pipeline()