import torch
from tqdm import tqdm
import torch.nn.functional as F

def validate(model, val_loader, device, criterion=None):
    model_to_eval = model.model if hasattr(model, 'model') else model
    criterion = criterion if criterion is not None else model.criterion
    
    model_to_eval.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            text_lengths = torch.tensor([len(text) for text in batch['text']]).to(device)
            
            outputs = model_to_eval(images)
            log_probs = F.log_softmax(outputs, dim=2).permute(1, 0, 2)

            input_lengths = torch.full(size=(outputs.size(0),), 
                                      fill_value=outputs.size(1),  
                                      dtype=torch.long,
                                      device=device)
            
            if (input_lengths < text_lengths).any():
                print("WARNING: input_lengths is smaller than text_lengths!")
                print("Problematic indices:", torch.where(input_lengths < text_lengths)[0])
            
            loss = criterion(log_probs, labels, input_lengths, text_lengths)
            total_loss += loss.item()
    
    return total_loss / num_batches

def evaluate_model(model_path, val_loader, vocab_size, device):
    """
    저장된 모델을 불러와 평가하는 함수
    
    Args:
        model_path: 체크포인트 경로
        val_loader: 검증 데이터 로더
        vocab_size: 어휘 크기
        device: 연산 장치
    """
    from models.crnn import OCRModel  # 여기서 import하여 순환 참조 방지
    
    model = OCRModel(vocab_size=vocab_size, device=device)
    model.load_checkpoint(model_path)
    
    val_loss = validate(
        model=model.model,
        val_loader=val_loader,
        device=device,
        criterion=model.criterion
    )
    
    # print(f'Validation Loss: {val_loss:.4f}')
    return val_loss

if __name__ == '__main__':
    from config import Config
    from datasets.ocr_dataset import create_dataloaders
    
    _, val_loader, vocab_size = create_dataloaders(
        root_dir=Config.DATA_ROOT,
        batch_size=Config.BATCH_SIZE
    )
    
    evaluate_model(
        model_path='checkpoints/best_model.pt',
        val_loader=val_loader,
        vocab_size=vocab_size,
        device=Config.DEVICE
    )