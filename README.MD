# Fool Image Generator

This project is focused on training a model to generate images specifically designed to fool image classifiers, based on the paper:  
**Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images**  
by Anh Nguyen, Jason Yosinski, and Jeff Clune.  
Paper link: [arXiv:1412.1897](https://arxiv.org/abs/1412.1897)

## Project Overview
Instead of using Evolution Algorithm, this repository use Neural Network with PyTorch Lightning-based training pipeline for generating images that are challenging for image classifiers to interpret correctly. 

## Requirements

To set up the environment, you can use the provided `requirements.txt` file.

### Step-by-Step Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/duong-nguy/foo-image.git
    cd fool-image-generator
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up Weights & Biases (optional)**:
    If you wish to log your training process to [Weights & Biases](https://wandb.ai/), ensure you have an API key. You can set this up by creating an account on W&B and then setting the environment variable:
    ```bash
    export WANDB_API_KEY=your_wandb_api_key
    ```

    **Note**: Logging is only enabled if you pass the `--log` flag when running the training script.

## Usage

The `train.py` script provides a command-line interface to train the fool image generator.

### Command-line Arguments

- `--batch_size`: The batch size for training.
- `--categories`: Number of categories (classes) for fooling the classifier.
- `--max_epochs`: Maximum number of epochs to train.
- `--duration`: Duration of training in `DD:HH:MM:SS` format (default: `00:04:50:00`).
- `--log`: Enables logging to Weights & Biases (optional).
- `--save_dir`: Directory where model checkpoints will be saved (default: `model_checkpoints`).
- `--ckpt_path`: Path to a checkpoint file to resume training (optional).

### Example Usage

To start training with logging enabled and a batch size of 32:

```bash
python train.py --batch_size 32 --categories 1000 --max_epochs 10 --duration 00:04:50:00 --log



