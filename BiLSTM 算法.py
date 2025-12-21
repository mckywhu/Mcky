import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import numpy as np

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理类
class TextPreprocessor:
    def __init__(self):
        pass
    
    def clean_text(self, text):
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        # 去除首尾空格
        text = text.strip()
        return text

# 词汇表类
class Vocabulary:
    def __init__(self, max_vocab_size=20000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()
        
    def build_vocab(self, texts):
        # 统计词频
        for text in texts:
            words = text.lower().split()
            self.word_freq.update(words)
        
        # 构建词汇表
        most_common = self.word_freq.most_common(self.max_vocab_size - 2)
        for i, (word, _) in enumerate(most_common):
            self.word2idx[word] = i + 2
            self.idx2word[i + 2] = word
            
    def text_to_sequence(self, text, max_length=100):
        words = text.lower().split()
        sequence = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        # 截断或填充序列
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence.extend([self.word2idx['<PAD>']] * (max_length - len(sequence)))
            
        return sequence

# 自定义数据集类
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 将文本转换为序列
        sequence = self.vocab.text_to_sequence(text)
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# BiLSTM模型
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.5):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向LSTM，所以hidden_dim*2
        
    def forward(self, x):
        # 词嵌入
        embedded = self.embedding(x)
        
        # BiLSTM层
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # 使用最后一个时间步的隐藏状态
        # 对于双向LSTM，连接前向和后向的最后隐藏状态
        final_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Dropout和全连接层
        output = self.dropout(final_hidden)
        output = self.fc(output)
        
        return output

# 模型训练类
class SentimentTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for texts, labels in self.train_loader:
            texts, labels = texts.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(texts)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for texts, labels in self.val_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                
                outputs = self.model(texts)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        accuracy = correct / total
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, accuracy, all_preds, all_labels

def load_training_data():
    """加载训练数据"""
    preprocessor = TextPreprocessor()
    
    # 读取正面评论
    with open('训练集/sample.positive.txt', 'r', encoding='utf-8') as f:
        content = f.read()
        # 解析XML格式的评论
        reviews = re.findall(r'<review id="[^"]*">([^<]*(?:<(?!/review>)[^<]*)*)</review>', content)
        positive_texts = [preprocessor.clean_text(review) for review in reviews if preprocessor.clean_text(review)]
    
    # 读取负面评论
    with open('训练集/sample.negative.txt', 'r', encoding='utf-8') as f:
        content = f.read()
        # 解析XML格式的评论
        reviews = re.findall(r'<review id="[^"]*">([^<]*(?:<(?!/review>)[^<]*)*)</review>', content)
        negative_texts = [preprocessor.clean_text(review) for review in reviews if preprocessor.clean_text(review)]
    
    # 创建标签
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    print(f"Positive samples: {len(positive_texts)}")
    print(f"Negative samples: {len(negative_texts)}")
    
    return texts, labels

def load_test_data():
    """加载测试数据"""
    preprocessor = TextPreprocessor()
    
    # 读取测试数据
    with open('测试集/test.en.txt', 'r', encoding='utf-8') as f:
        test_content = f.read()
        test_reviews = re.findall(r'<review id="[^"]*">([^<]*(?:<(?!/review>)[^<]*)*)</review>', test_content)
        test_texts = [preprocessor.clean_text(review) for review in test_reviews]
    
    # 读取测试标签
    with open('测试集标注/test.label.en.txt', 'r', encoding='utf-8', errors='ignore') as f:
        label_content = f.read()
        # 提取标签
        test_labels = []
        labels_matches = re.findall(r'label="(\d)"', label_content)
        for label in labels_matches:
            test_labels.append(int(label))
    
    return test_texts, test_labels

def main():
    # 数据预处理
    print("Loading training data...")
    texts, labels = load_training_data()
    
    # 构建词汇表
    print("Building vocabulary...")
    vocab = Vocabulary(max_vocab_size=20000)
    vocab.build_vocab(texts)
    
    # 划分训练集和验证集 (80%训练, 20%验证)
    split_idx = int(0.8 * len(texts))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # 创建数据集和数据加载器
    train_dataset = SentimentDataset(train_texts, train_labels, vocab)
    val_dataset = SentimentDataset(val_texts, val_labels, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型
    print("Initializing model...")
    model = BiLSTMClassifier(
        vocab_size=len(vocab.word2idx),
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=2,
        dropout=0.5
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    trainer = SentimentTrainer(model, train_loader, val_loader, criterion, optimizer, device)
    
    print("Training model...")
    num_epochs = 5
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch()
        val_loss, val_acc, _, _ = trainer.evaluate()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 在测试集上评估
    print("Evaluating on test set...")
    test_texts, test_labels = load_test_data()
    
    # 创建测试数据集
    test_dataset = SentimentDataset(test_texts, test_labels, vocab)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 评估测试集
    model.eval()
    test_preds = []
    test_true = []
    test_loss = 0
    
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader)
    
    # 输出结果到文件1 (按TeamN_1_EN.txt格式)
    print("Generating results file 1...")
    with open('结果/result.txt', 'w', encoding='utf-8') as f:
        for i, pred in enumerate(test_preds):
            label_str = "positive" if pred == 1 else "negative"
            f.write(f"TeamN 1 {i+1} {label_str}\n")
    
    # 输出结果到文件2 (包含测试损失、分类报告和混淆矩阵)
    print("Generating results file 2...")
    with open('结果/evaluation_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        
        # 分类报告
        f.write("Classification Report:\n")
        report = classification_report(test_true, test_preds, target_names=['Negative', 'Positive'], digits=4)
        f.write(report)
        f.write("\n\n")
        
        # 混淆矩阵
        f.write("Confusion Matrix:\n")
        cm = confusion_matrix(test_true, test_preds)
        f.write(str(cm))
        f.write("\n")
    
    print("Results saved to result.txt and evaluation_results.txt")
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
