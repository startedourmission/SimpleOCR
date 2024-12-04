import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(CRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2,1))
        )
        
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, vocab_size + 1)
        
    def forward(self, x):

        conv = self.cnn(x)

        b, c, h, w = conv.size()
        conv = conv.squeeze(2)  

        conv = conv.permute(0, 2, 1)  

        rnn_out, _ = self.rnn(conv)

        output = self.fc(rnn_out)

        return output

class OCRModel:
    def __init__(self, vocab_size, device='cuda'):
        self.device = device
        self.model = CRNN(vocab_size).to(device)
        self.criterion = nn.CTCLoss(blank=0, reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
    
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])