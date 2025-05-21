.PHONY: dev install train clean

# Activate venv and run interactive shell
dev:
	source .venv/bin/activate && bash

# Install dependencies from requirements.txt
install:
	source .venv/bin/activate && pip install -r requirements.txt

# Train model (assumes train.py exists)
train:
	source .venv/bin/activate && python scripts/train.py

# Reset environment
clean:
	rm -rf __pycache__ *.pyc *.log results/model/ results/logs/ 