#!/bin/bash
set -e

# Spinner function
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\\'
    while ps -p $pid >/dev/null 2>&1; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
}

# Parse CLI args
GIT_SETUP=true

for arg in "$@"; do
    case $arg in
    --model=*)
        MODEL_NAME="${arg#*=}"
        shift
        ;;
    --labels=*)
        NUM_LABELS="${arg#*=}"
        shift
        ;;
    --git-setup=*)
        GIT_SETUP="${arg#*=}"
        shift
        ;;
    esac
done

echo -n "🔧 Updating system packages..."
(sudo apt update -y && sudo apt install -y git python3-pip python3-venv) >/dev/null 2>&1 &
spinner $!
echo " ✅"

if [ "$GIT_SETUP" != "false" ]; then
    GIT_EMAIL=$(git config --global user.email || true)
    GIT_NAME=$(git config --global user.name || true)

    if [ -z "$GIT_EMAIL" ] || [ -z "$GIT_NAME" ]; then
        echo "🔐 Git global config not found. Let's set it up."

        read -p "📧 Enter your Git email: " GIT_EMAIL
        read -p "👤 Enter your Git name: " GIT_NAME

        git config --global user.email "$GIT_EMAIL"
        git config --global user.name "$GIT_NAME"

        echo "✅ Git has been configured globally."
    else
        echo "✅ Git global config already set: $GIT_NAME <$GIT_EMAIL>"
    fi
else
    echo "🚫 Git setup skipped via --git-setup=false"
fi

echo "🗂️  Creating project folders (if missing)..."
mkdir -p data/working/{original,checkpoint,training} models results config scripts

echo -n "🐍 Creating virtual environment (if needed)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv >/dev/null 2>&1
fi
echo " ✅"

echo -n "📦 Activating virtual environment..."
source .venv/bin/activate
echo " ✅"

echo -n "⚡ Installing PyTorch with CUDA (11.8)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 >/dev/null 2>&1 &
spinner $!
echo " ✅"

echo -n "📚 Installing Transformers & training dependencies..."
pip install transformers datasets accelerate evaluate wandb pyyaml scikit-learn >/dev/null 2>&1 &
spinner $!
echo " ✅"

echo -n "🛠️  Installing development tools..."
pip install black isort pylint >/dev/null 2>&1 &
spinner $!
echo " ✅"

if [ ! -f "requirements.txt" ]; then
    echo -n "📝 Freezing requirements.txt..."
    pip freeze >requirements.txt
    echo " ✅"
fi

echo -n "🔓 Making run.sh executable (if present)..."
[ -f run.sh ] && chmod +x run.sh && echo " ✅" || echo " ⚠️  Skipped (run.sh not found)"

ACCEL_CFG="$HOME/.cache/huggingface/accelerate/default_config.yaml"

if [ -f "$ACCEL_CFG" ]; then
    echo "⚙️  Accelerate is already configured. Skipping interactive setup."
else
    echo "⚙️  Launching Accelerate configuration (interactive)..."
    accelerate config
fi

CONFIG_PATH="config/training_config.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "🧠 Initializing training config..."

    if [ -z "$MODEL_NAME" ]; then
        read -p "🤖 Enter Hugging Face model name [default: distilbert-base-uncased]: " MODEL_NAME
        MODEL_NAME=${MODEL_NAME:-distilbert-base-uncased}
    fi

    if [ -z "$NUM_LABELS" ]; then
        read -p "🎯 Enter number of output labels [default: 2]: " NUM_LABELS
        NUM_LABELS=${NUM_LABELS:-2}
    fi

    mkdir -p config

    cat <<EOF >$CONFIG_PATH
model:
  name: "$MODEL_NAME"
  num_labels: $NUM_LABELS

data:
  train_file: "data/train.json"
  validation_file: "data/validation.json"
  text_column: "text"
  label_column: "label"
  max_length: 128

training:
  output_dir: "results/model"
  num_train_epochs: 3
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 64
  warmup_steps: 500
  weight_decay: 0.01
  logging_dir: "results/logs"
  logging_steps: 100
  evaluation_strategy: "epoch"
  save_strategy: "epoch"
  seed: 42
  fp16: true
  gradient_accumulation_steps: 2
  learning_rate: 5e-5
  lr_scheduler_type: "linear"
  max_grad_norm: 1.0
EOF

    echo "✅ Created config/training_config.yaml"
else
    echo "📄 config/training_config.yaml already exists — skipping config setup."
fi

echo ""
echo "✅ Environment setup complete!"
echo "📂 To activate your environment later, run:"
echo "   source .venv/bin/activate"
echo "🚀 Then run './run.sh' or 'make train' to begin training."
