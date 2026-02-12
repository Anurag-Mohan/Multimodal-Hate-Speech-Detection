import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize



BATCH_SIZE = 16          
LEARNING_RATE = 0.001
EPOCHS = 5               
EMBED_DIM = 100          
HIDDEN_DIM = 64          
MAX_LEN = 50             
VOCAB_SIZE = 5000        
IMG_SIZE = 128           


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


nltk.download('punkt', quiet=True)



def build_vocab(texts):
    """Creates a dictionary to map words to numbers."""
    all_words = []
    for text in texts:
        all_words.extend(word_tokenize(str(text).lower()))
    count = Counter(all_words)
    
    vocab = {word: i+1 for i, (word, _) in enumerate(count.most_common(VOCAB_SIZE))}
    return vocab

def text_pipeline(text, vocab):
    """Converts a text string into a list of numbers (padded/truncated)."""
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        
        img_path = self.data.iloc[idx]['img']  
        
        try:
            
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            
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
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        
        
        
        self.fc = nn.Linear(256 * 8 * 8, 128) 

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) 
        x = self.pool(self.relu(self.conv2(x))) 
        x = self.pool(self.relu(self.conv3(x))) 
        x = self.pool(self.relu(self.conv4(x))) 
        x = self.flatten(x)
        x = self.relu(self.fc(x))
        return x


class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(TextBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(hidden_dim * 2, 128) 

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden_cat)


class MultimodalNet(nn.Module):
    def __init__(self, vocab_size):
        super(MultimodalNet, self).__init__()
        self.cnn = CustomCNN()
        self.lstm = TextBiLSTM(vocab_size, EMBED_DIM, HIDDEN_DIM)
        
        
        self.fusion = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(64, 1),
            nn.Sigmoid()     
        )

    def forward(self, image, text):
        visual_feat = self.cnn(image)
        text_feat = self.lstm(text)
        
        combined = torch.cat((visual_feat, text_feat), dim=1)
        output = self.fusion(combined)
        return output


def train_model():
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("--- Loading Dataset ---")
    try:
        
        df = pd.read_json('train.jsonl', lines=True)
        vocab = build_vocab(df['text'].tolist())
        print(f"Vocabulary built with {len(vocab)} words.")

        
        dataset = MemeDataset('train.jsonl', vocab, transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"Data loaded: {len(dataset)} memes found.")
        
    except FileNotFoundError:
        print("CRITICAL ERROR: 'train.jsonl' not found.")
        print("Please ensure train.py and train.jsonl are in the same folder.")
        return
    except ValueError as e:
        print("JSONL Format Error:", e)
        return

    
    model = MultimodalNet(len(vocab)).to(device)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"--- Starting Training for {EPOCHS} Epochs ---")
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (imgs, texts, labels) in enumerate(loader):
            
            imgs, texts, labels = imgs.to(device), texts.to(device), labels.to(device)
            labels = labels.unsqueeze(1) 

            
            outputs = model(imgs, texts)
            loss = criterion(outputs, labels)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            
            
            if i % 100 == 0 and i > 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}], Loss: {loss.item():.4f}")

        epoch_acc = 100 * correct / total
        print(f"=== Epoch {epoch+1} Finished. Accuracy: {epoch_acc:.2f}% ===")

    
    torch.save(model.state_dict(), 'meme_model.pth')
    
    
    import pickle
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
        
    print("\nSUCCESS! Model saved as 'meme_model.pth' and Vocab as 'vocab.pkl'")

if __name__ == "__main__":
    train_model()