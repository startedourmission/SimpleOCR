class Config:

    # DATA_ROOT = "/scratch/e1430a12/workspace/OCR/71299_subset"
    DATA_ROOT = "/scratch/e1430a12/workspace/OCR/71299_sample"
    BATCH_SIZE = 16
    NUM_WORKERS = 4

    HIDDEN_SIZE = 256
    LEARNING_RATE = 0.00001

    NUM_EPOCHS = 100
    DEVICE = "cuda"  # or "cpu"
    CHECKPOINT_DIR = "checkpoints"

    LOG_INTERVAL = 100
    SAVE_INTERVAL = 1