from pathlib import Path

repo_dir = Path(__file__).parent.parent

weights_path = repo_dir / 'model_75.pt'

# Paths
DATA_DIR = Path(repo_dir / "exampleData/")
TRAIN_DIR = DATA_DIR / "trainingData"
VAL_DIR = DATA_DIR / "validationData"
OUT_DIR = Path(repo_dir / "models/")

# Training Parameters
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 2
NUM_EPOCHS = 500
LEARNING_RATE = 0.001
SAVE_INTERVAL = 5  # Save model every 5 epochs
