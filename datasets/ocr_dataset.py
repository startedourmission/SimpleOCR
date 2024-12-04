import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from pathlib import Path
import numpy as np
import glob

class OCRDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        self.image_dir = self.root_dir / split / 'images'
        self.label_dir = self.root_dir / split / 'labels'

        self.image_files = sorted([Path(f).name for f in glob.glob(os.path.join(self.image_dir, '*.*'))])
        self.label_files = sorted([Path(f).name for f in glob.glob(os.path.join(self.label_dir, '*.json'))])
        
        self.char_to_idx = self._create_vocab()
        
    def _create_vocab(self):

        chars = set()
        for label_file in self.label_files:
            label_path = self.label_dir / label_file
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    annotation = json.load(f)
                    for bbox in annotation['Bbox']:
                        chars.update(list(bbox['data']))
            except Exception as e:
                print(f"Error processing {label_file}: {e}")
                continue
        
        vocabulary = ["-"] + sorted(list(chars))  
        
        return {char: idx for idx, char in enumerate(vocabulary)}
    
    def _encode_text(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char_to_idx[c] for c in text])

    
    def _extract_text_region(self, image: Image.Image, bbox: dict) -> Image.Image:
        x_coords = bbox['x']
        y_coords = bbox['y']
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.width, x_max)
        y_max = min(image.height, y_max)
        
        try:
            text_region = image.crop((x_min, y_min, x_max, y_max))
            return text_region
        except Exception as e:
            print(f"Error cropping image: {e}")
            return image
    
    def __len__(self):
        return len(self.label_files)
    
    def __getitem__(self, idx):
        try:
            label_path = self.label_dir / self.label_files[idx]
            with open(label_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            
            image_file = self.image_files[idx]
            image_path = self.image_dir / image_file
            image = Image.open(image_path).convert('RGB')
            
            bbox = np.random.choice(annotation['Bbox'])
            text_region = self._extract_text_region(image, bbox)
            
            if self.transform:
                text_region = self.transform(text_region)
            
            label = self._encode_text(bbox['data'])
            
            return {
                'image': text_region,
                'text': bbox['data'],
                'label': label,
                'bbox_type': bbox['type'],
                'typeface': bbox['typeface'],
                'file_name': image_file
            }
            
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return {
                'image': torch.zeros((1, 32, 64)),
                'text': '',
                'label': torch.tensor([0]),
                'bbox_type': 0,
                'typeface': 0,
                'file_name': 'error.jpg'
            }


def collate_fn(batch):

    images = []
    labels = []
    texts = []
    bbox_types = []
    typefaces = []
    file_names = []
    
    for item in batch:
        images.append(item['image'])
        labels.append(item['label'])
        texts.append(item['text'])
        bbox_types.append(item['bbox_type'])
        typefaces.append(item['typeface'])
        file_names.append(item['file_name'])

    images = torch.stack(images)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    return {
        'image': images,
        'label': labels,
        'text': texts,
        'bbox_type': torch.tensor(bbox_types),
        'typeface': torch.tensor(typefaces),
        'file_name': file_names
    }


def create_dataloaders(root_dir: str, batch_size: int = 32):

    transform = transforms.Compose([
        transforms.Resize((21, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = OCRDataset(root_dir=root_dir, split='train', transform=transform)
    val_dataset = OCRDataset(root_dir=root_dir, split='val', transform=transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True  
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn 
    )
    
    vocab_size = len(train_dataset.char_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    max_idx = max(train_dataset.char_to_idx.values())
    print(f"Maximum index in vocabulary: {max_idx}")
    
    return train_loader, val_loader, vocab_size
