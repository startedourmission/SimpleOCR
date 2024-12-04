import os
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from config import Config
from models.crnn import OCRModel
from datasets.ocr_dataset import create_dataloaders
from evaluate import validate
import shutil 
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

def decode_predictions(outputs, idx_to_char):
    """모델 출력을 텍스트로 디코딩"""
    pred_tokens = F.softmax(outputs, 2).argmax(2)  # [T, batch_size]
    pred_tokens = pred_tokens.transpose(0, 1)  # [batch_size, T]
    
    result = []
    for tokens in pred_tokens:
        text = ""
        prev_token = None
        for token in tokens:
            token = token.item()
            if token != 0 and token != prev_token:  # 0 is blank token
                text += idx_to_char.get(token, '')
            prev_token = token
        result.append(text)
    return result

def calculate_metrics(pred_texts, true_texts):
    """정확도 메트릭 계산"""
    metrics = defaultdict(float)
    total = len(true_texts)
    
    for pred, true in zip(pred_texts, true_texts):
        # Exact match
        if pred == true:
            metrics['exact_match'] += 1
            
        # Character-level metrics
        true_chars = set(true)
        pred_chars = set(pred)
        
        tp = len(true_chars & pred_chars)
        fp = len(pred_chars - true_chars)
        fn = len(true_chars - pred_chars)
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics['char_precision'] += precision
        metrics['char_recall'] += recall
    
    # 평균 계산
    metrics = {k: v/total for k, v in metrics.items()}
    
    # F1 계산
    if metrics['char_precision'] + metrics['char_recall'] > 0:
        metrics['char_f1'] = 2 * (metrics['char_precision'] * metrics['char_recall']) / (metrics['char_precision'] + metrics['char_recall'])
    else:
        metrics['char_f1'] = 0.0
        
    return metrics

def evaluate_model(model, data_loader, device, idx_to_char):
    """검증 데이터에 대한 평가"""
    model.eval()
    metrics_sum = defaultdict(float)
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            true_texts = batch['text']
            
            outputs = model(images)
            pred_texts = decode_predictions(outputs, idx_to_char)
            
            batch_metrics = calculate_metrics(pred_texts, true_texts)
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
    
    # 전체 평균 계산
    num_batches = len(data_loader)
    metrics_avg = {k: v/num_batches for k, v in metrics_sum.items()}
    return metrics_avg

def visualize_predictions(model, batch, char_to_idx, idx_to_char, save_dir, epoch, device):
    model.eval()
    with torch.no_grad():
        images = batch['image'].to(device)
        texts = batch['text']
        outputs = model(images)
        
        # 예측 디코딩
        pred_tokens = F.softmax(outputs, 2).argmax(2)  # [T, batch_size]
        pred_tokens = pred_tokens.transpose(0, 1)  # [batch_size, T]
        
        # 시각화할 이미지 수 (최대 4개)
        num_vis = min(4, len(images))
        
        for i in range(num_vis):
            img = images[i].cpu().squeeze(0)
            img = ((img * 0.5 + 0.5) * 255).byte().numpy()
            img = Image.fromarray(img).convert('RGB')
            
            # 예측 텍스트 생성
            pred_text = ""
            prev_token = None
            for token in pred_tokens[i]:
                token = token.item()
                if token != 0 and token != prev_token:
                    pred_text += idx_to_char[token]
                prev_token = token
            
            # 이미지 크기 2배로 키워서 텍스트가 잘 보이게
            img = img.resize((img.width * 2, img.height * 2))
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("../NanumGothic.ttf", 30)  # 폰트 크기도 키움
            
            # GT와 Pred 모두 표시 (y 위치 간격 늘림)
            draw.text((10, 10), f'Text: {texts[i]}', fill='blue', font=font)
            draw.text((10, 40), f'Pred: {pred_text}', fill='red', font=font)  
            
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'epoch_{epoch}_sample_{i}.png')
            img.save(save_path)
    
    model.train()
    
def compute_loss(text_batch, text_batch_logits, criterion, char_to_idx, device):
    text_batch_logits = text_batch_logits.permute(1, 0, 2) 
    text_batch_logps = F.log_softmax(text_batch_logits, 2)
    
    batch_size = len(text_batch)
    
    targets = []
    target_lengths = torch.zeros(batch_size, dtype=torch.int32).to(device)
    
    for i, text in enumerate(text_batch):
        text_indices = [char_to_idx[c] for c in text]
        targets.extend(text_indices)
        target_lengths[i] = len(text_indices)
    
    targets = torch.tensor(targets, dtype=torch.int32).to(device)
    input_lengths = torch.full(size=(batch_size,), 
                             fill_value=text_batch_logits.size(0),
                             dtype=torch.int32).to(device)
    
    return criterion(text_batch_logps, targets, input_lengths, target_lengths)

def train(config):

    train_loader, val_loader, vocab_size = create_dataloaders(
        root_dir=config.DATA_ROOT,
        batch_size=config.BATCH_SIZE
    )
    

    device = torch.device(config.DEVICE)
    model = OCRModel(vocab_size=vocab_size, device=device)
    model.dataset = train_loader.dataset 
    

    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)

    idx_to_char = {idx: char for char, idx in train_loader.dataset.char_to_idx.items()}

    best_val_loss = float('inf')
    best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')
    
    vis_dir = os.path.join(config.CHECKPOINT_DIR, 'visualizations')
    Path(vis_dir).mkdir(exist_ok=True, parents=True)
    
    # 학습 루프
    for epoch in range(config.NUM_EPOCHS):
        model.model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            texts = batch['text']
            
            model.optimizer.zero_grad()
            outputs = model.model(images)
            
            loss = compute_loss(texts, outputs, model.criterion, train_loader.dataset.char_to_idx, device)
            loss.backward()
            model.optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % (config.LOG_INTERVAL * 5) == 0:  # LOG_INTERVAL의 5배마다 시각화
                visualize_predictions(
                    model=model.model,
                    batch=batch,
                    char_to_idx=train_loader.dataset.char_to_idx,
                    idx_to_char={idx: char for char, idx in train_loader.dataset.char_to_idx.items()},
                    save_dir=vis_dir,
                    epoch=epoch,
                    device=device
                )
                
                
            if batch_idx % config.LOG_INTERVAL == 0:
                pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        val_loss = validate(model, val_loader, device)
        val_metrics = evaluate_model(model.model, val_loader, device, idx_to_char)
        
        print(f'\nEpoch {epoch+1}/{config.NUM_EPOCHS}:')
        print(f'Average Train Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Metrics:')
        print(f'- Char Precision: {val_metrics["char_precision"]*100:.2f}%')
        print(f'- Char Recall: {val_metrics["char_recall"]*100:.2f}%')
        print(f'- Char F1: {val_metrics["char_f1"]*100:.2f}%')
        
        print(f'Average Train Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(best_model_path)
            print(f'New best model saved with validation loss: {val_loss:.4f}')
            
        

if __name__ == '__main__':
    train(Config)