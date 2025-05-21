# HF Transformers Training Boilerplate

A boilerplate project structure for training Hugging Face Transformer models on WSL2.

## Project Structure

```
training-boilerplate/
├── .venv/                     # Auto-created, ignored in Git
├── data/                      # Drop in your training data here
│   ├── train.json
│   └── working/               # Structured working data lifecycle
│       ├── original/          # Raw downloaded datasets
│       ├── checkpoint/        # Manually cleaned, edited versions
│       └── complete/          # Finalized training-ready versions
├── models/                    # Optional: custom model configs or checkpoints
├── results/                   # Output directory (checkpoints, logs, etc.)
├── config/
│   └── training_config.yaml   # Training configuration
├── scripts/
│   ├── train.py               # Core training script
│   └── preprocess.py          # Data preparation script
├── Makefile                   # Task runner
├── run.sh                     # Simple venv + train launcher
├── setup.sh                   # First-time setup with optional CLI flags
└── README.md                  # This file
```

## Getting Started

### 🌟 First-time Setup

Run the setup script to create a virtual environment and install dependencies:

```bash
chmod +x setup.sh
./setup.sh [--model=MODEL_NAME] [--labels=NUM_LABELS]
```

This will:
- Create necessary directories
- Set up a Python virtual environment
- Install PyTorch with CUDA support
- Install Transformers and other dependencies
- Generate a training config interactively or via flags
- Freeze `requirements.txt`

### Data Preparation

Place your raw data in any common format (CSV, JSON, etc.) in the `data/` directory, then use the preprocessing script to prepare it for training:

```bash
source .venv/bin/activate
python scripts/preprocess.py --input_file path/to/your/data.csv --text_column text --label_column label
```

This will:
- Load your dataset
- Analyze text lengths to recommend optimal sequence length
- Split data into train/validation sets
- Save processed data to the data/ directory in the proper format

### Configuration

Edit `config/training_config.yaml` to customize:
- Model selection
- Data paths and columns
- Training parameters (batch size, learning rate, etc.)
- Optimization settings (mixed precision, gradient accumulation, etc.)
- Optional distributed settings (DeepSpeed, DDP)

### Training

Start training using one of these methods:

**Simple way:**
```bash
./run.sh
```

**Using the Makefile:**
```bash
make train
```

**Direct invocation with custom parameters:**
```bash
source .venv/bin/activate
python scripts/train.py --model_name google/bert_uncased_L-4_H-256_A-4 --data_path data/custom_data.json
```

## Customization

### Using Different Models

To use a different model, either:
- Edit the `model.name` field in `config/training_config.yaml`, or
- Pass it directly:

```bash
python scripts/train.py --model_name facebook/bart-base
```

### Distributed Training

The training script supports distributed training with PyTorch. For multi-GPU training:

```bash
python -m torch.distributed.run --nproc_per_node=NUM_GPUS scripts/train.py
```

### Weights & Biases Integration

To use Weights & Biases for experiment tracking:

1. Install wandb: `pip install wandb`
2. Log in: `wandb login`
3. Set your API key: `export WANDB_API_KEY=your_api_key`

The training script will automatically log metrics if an API key is detected.

## Management Commands

```bash
# Enter dev environment
make dev

# Install dependencies
make install

# Run training
make train

# Clean up project
make clean
```

## Requirements

- WSL2 with Ubuntu 22.04 or similar
- Python 3.8+
- CUDA support (for GPU training)

## License

This project is open source and available under the MIT License.
