한글 OCR 프로젝트

CRNN(CNN + RNN + CTC Loss) 구조를 활용한 한글 텍스트 인식 모델 구현 프로젝트입니다.

## 📌 프로젝트 소개

- **모델 구조**: CRNN (CNN + RNN + CTC Loss)
- **데이터셋 규모**: 
  - 학습 데이터: 5만개
  - 검증 데이터: 5천개
- **텍스트 특성**:
  - 평균 길이: 3.62 (범위: 1-37)
  - 고유 문자: 921개 (숫자, 구두점, 한글 혼합)

## 🔨 모델 구조

### CNN 모듈
```python
CNN 구조:
- Conv2d(1, 64, 3) + ReLU + MaxPool2d
- Conv2d(64, 128, 3) + ReLU + MaxPool2d
- Conv2d(128, 256, 3) + BatchNorm + ReLU + MaxPool2d
- Conv2d(256, 512, 3) + BatchNorm + ReLU
- Conv2d(512, 512, 3) + BatchNorm + ReLU + MaxPool2d
```

### RNN 모듈
- 양방향 LSTM 사용
- 2개 층 구성
- 은닉층 크기: 256
- 입력 크기: 512 (CNN에서 출력)

## 📁 프로젝트 구조

```
.
├── config.py           # 설정 파일
├── models/
│   └── crnn.py        # CRNN 모델 구현
├── datasets/
│   └── ocr_dataset.py # 데이터셋 및 데이터로더 구현
├── train.py           # 학습 스크립트
├── evaluate.py        # 평가 스크립트
└── inference.py       # 추론 및 시각화
```

## ⚙️ 설정

`config.py` 주요 설정:
```python
BATCH_SIZE = 16
HIDDEN_SIZE = 256
LEARNING_RATE = 0.00001
NUM_EPOCHS = 100
```

## 🔄 데이터 전처리

- 이미지 크기 조정: 21x256 픽셀
- 그레이스케일 변환
- 정규화 (mean=0.5, std=0.5)
- 학습 시 무작위 bbox 선택
- 가변 길이 텍스트 입력 처리를 위한 collate 함수 구현

## 🚀 실행 방법

### 학습
```bash
python train.py
```

### 평가
```bash
python evaluate.py
```

### 추론
```bash
python inference.py
```

## 🔍 현재 이슈 및 해결 방안

### 학습 불안정 문제:
1. 학습/검증 손실이 불규칙하게 변동
2. 배치별 최대 라벨 값 변동 (537-561 범위)
3. 배치별 라벨 형태 불일치 ([32, 8] ~ [32, 16])
4. 간헐적인 무한대 학습 손실 발생

### 해결 방안:
1. vocab mapping 로직 재검토 및 표준화
2. 라벨 인덱싱 일관성 확보
3. 배치간 라벨 형태 통일화

## ✅ 할 일

- [ ] 배치간 vocab mapping 안정화
- [ ] 일관된 라벨 인덱싱 구현
- [ ] 배치간 라벨 형태 표준화
- [ ] 데이터 증강 기법 추가
- [ ] early stopping 구현
- [ ] 모델 체크포인팅 추가

## 📊 성능 지표

현재 측정 지표:
- 학습 손실
- 검증 손실
- 문자 오류율 (구현 예정)
- 단어 오류율 (구현 예정)

## 🛠 개발 환경

- PyTorch
- torchvision
- Pillow
- numpy
- tqdm

## 📋 참고사항

- `requirements.txt` 설치 필요
- CUDA 지원 GPU 권장
- 한글 폰트 파일 필요 (`NanumGothic.ttf`)

