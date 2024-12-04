import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import os
from pathlib import Path
from config import Config
from models.crnn import OCRModel
from datasets.ocr_dataset import OCRDataset, create_dataloaders
from tqdm import tqdm
from collections import defaultdict

class OCRPredictor:
    def __init__(self, model_path, dataset_path, device='cuda'):
        self.device = torch.device(device)
        
        # 학습 데이터셋의 vocab 로드
        train_dataset = OCRDataset(root_dir=dataset_path, split='train')
        self.char_to_idx = train_dataset.char_to_idx
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # 모델 초기화 및 로드
        self.model = OCRModel(vocab_size=self.vocab_size, device=self.device)
        self.model.load_checkpoint(model_path)
        self.model.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((21, 256)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def decode_predictions(self, log_probs):
        """Decode model predictions to text"""
        pred_tokens = F.softmax(log_probs, 2).argmax(2)  # [T, batch_size]
        pred_tokens = pred_tokens.transpose(0, 1)  # [batch_size, T]
        
        result = []
        for tokens in pred_tokens:
            text = ""
            prev_token = None
            for token in tokens:
                token = token.item()
                if token != 0 and token != prev_token:  # 0 is blank token
                    text += self.idx_to_char[token]
                prev_token = token
            result.append(text)
        return result

    def calculate_metrics(self, pred_text, true_text):
        """Calculate various accuracy metrics"""
        # Exact match
        exact_match = pred_text == true_text
        
        # Character-level metrics
        true_chars = set(true_text)
        pred_chars = set(pred_text)
        
        # True positives (correctly predicted characters)
        tp = len(true_chars & pred_chars)
        # False positives (predicted but not in true)
        fp = len(pred_chars - true_chars)
        # False negatives (in true but not predicted)
        fn = len(true_chars - pred_chars)
        
        # Character-level precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        
        return {
            'exact_match': exact_match,
            'char_precision': precision,
            'char_recall': recall,
            'char_f1': f1,
        }
    
    def evaluate_dataset(self, val_loader):
        """Evaluate model on entire dataset"""
        metrics_sum = defaultdict(float)
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                true_texts = batch['text']
                
                outputs = self.model.model(images)
                pred_texts = self.decode_predictions(outputs)
                
                for pred_text, true_text in zip(pred_texts, true_texts):
                    metrics = self.calculate_metrics(pred_text, true_text)
                    for k, v in metrics.items():
                        metrics_sum[k] += v
                    total_samples += 1
        
        # Calculate averages
        metrics_avg = {k: v/total_samples for k, v in metrics_sum.items()}
        return metrics_avg

def main():
    predictor = OCRPredictor(
        model_path="checkpoints/best_model.pt",
        dataset_path=Config.DATA_ROOT,
        device=Config.DEVICE
    )
    
    # 전체 검증 데이터셋에 대한 평가
    _, val_loader, _ = create_dataloaders(
        root_dir=Config.DATA_ROOT,
        batch_size=Config.BATCH_SIZE
    )
    
    print("\nEvaluating on validation dataset...")
    metrics = predictor.evaluate_dataset(val_loader)
    
    print("\nOverall Metrics:")
    print(f"Exact Match Accuracy: {metrics['exact_match']*100:.2f}%")
    print(f"Character Precision: {metrics['char_precision']*100:.2f}%")
    print(f"Character Recall: {metrics['char_recall']*100:.2f}%")
    print(f"Character F1 Score: {metrics['char_f1']*100:.2f}%")
    
    return metrics

if __name__ == '__main__':
    main()