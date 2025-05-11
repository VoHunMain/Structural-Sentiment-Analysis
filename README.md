# Structured Sentiment Analysis

## Overview

This repository implements a comprehensive neural architecture for structured sentiment analysis in PyTorch. The model simultaneously extracts sentiment holders, targets, and expressions while classifying sentiment polarity and intensity, providing a complete analysis of sentiment structures within text.

Structured sentiment analysis goes beyond traditional sentiment classification by identifying:
- **Who** is expressing the sentiment (holder)
- **What** the sentiment is about (target)
- **How** the sentiment is expressed (expression)
- **What type** of sentiment is expressed (polarity and intensity)

## Features

- 🌐 **Multi-lingual Support**: Built on XLM-RoBERTa for cross-lingual transfer learning
- 🔍 **Span Detection**: Identifies sentiment holders, targets, and expressions with BIO tagging
- 🔄 **Relation Classification**: Models relationships between detected spans
- 🎭 **Polarity Classification**: Positive, negative, neutral sentiment classification
- 📊 **Intensity Classification**: Strong, average, weak intensity determination
- 🔌 **Optional Language Adapters**: Efficient multi-lingual fine-tuning with minimal parameters

## Model Architecture

![Model Architecture](https://via.placeholder.com/800x400?text=Structured+Sentiment+Analysis+Architecture)
![Model Architecture](https://via.placeholder.com/800x400?raw=true)
The model consists of several components:

1. **Base Encoder**: XLM-RoBERTa produces contextualized token representations
2. **Span Attention**: Self-attention layers enhance span-aware representations
3. **Span Detectors**: Three parallel classifiers identify holder, target, and expression spans
4. **Cross-Span Attention**: Models relationships between detected spans
5. **Classifiers**: Separate modules for relation, polarity, and intensity classification

## Requirements

```
torch>=1.9.0
transformers>=4.18.0
numpy>=1.20.0
tqdm>=4.62.0
scikit-learn>=1.0.0
```

## Installation

```bash
git clone https://github.com/yourusername/structured-sentiment.git
cd structured-sentiment
pip install -r requirements.txt
```

## Data Format

The model expects JSON data in the following format:

```json
[
  {
    "text": "I love this product.",
    "sent_id": "example-1",
    "opinions": [
      {
        "Source": "0:1",
        "Target": "7:14",
        "Polar_expression": "2:6",
        "Polarity": "Positive",
        "Intensity": "Strong"
      }
    ]
  }
]
```

Where span indices (e.g., "0:1") represent character offsets in the text.

## Usage
The .ipynb file has been submitted and the drive link for the model checkpoints for all different datasets have been submitted. Now to run the code in google colab, you will have to download the folder in your personal drive, and then before running the training file you will have to dock it in your google colab by using the command below:

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

Now after mounting it, just copy the paths of the train, dev and test datasets on whch the model has to be trained are to be pasted in the main funciton in the lines as explained below:

```python
train_file = os.path.join(data_dir, language, "/content/train.json")
dev_file = os.path.join(data_dir, language, "/content/dev.json")
test_file = os.path.join(data_dir, language, "/content/test.json")
```

### Training

```python
from transformers import XLMRobertaTokenizerFast
from torch.utils.data import DataLoader
import torch.optim as optim

# Initialize tokenizer and datasets
tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
train_dataset = SentimentDataset("train.json", tokenizer)
dev_dataset = SentimentDataset("dev.json", tokenizer)

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=8, 
    shuffle=True, 
    collate_fn=custom_collate
)
dev_dataloader = DataLoader(
    dev_dataset, 
    batch_size=8, 
    shuffle=False, 
    collate_fn=custom_collate
)

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StructuredSentimentModel().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, 
    num_training_steps=len(train_dataloader)*5
)

# Train the model
train_model(
    model, 
    train_dataloader, 
    dev_dataloader, 
    optimizer, 
    scheduler, 
    device, 
    num_epochs=5, 
    output_dir="./models"
)
```

### Evaluation

```python
test_dataset = SentimentDataset("test.json", tokenizer)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=8, 
    shuffle=False, 
    collate_fn=custom_collate
)
test_results = evaluate_model(model, test_dataloader, device)

# Print results
for span_type, metrics in test_results.items():
    print(f"{span_type}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
```

### Inference

In the inference.ipynb file, the code can be run where you can input a sentence and just paste the path of the model you want to use from the mounted google drive, and just hit RUN!! It will give you the predicted tuples , the tokenised input sentence and the XLM-RoBERTa encoded embeddings.

In the last code in this file, if you run it it will ask you to upload a test.json file of your wish and then it will output a file having all the predicted sentiments and also the precision, recall and F1 scores.


## Configuration

Key hyperparameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MAX_SEQ_LENGTH` | Maximum sequence length | 512 |
| `NUM_EPOCHS` | Number of training epochs | 5 |
| `BATCH_SIZE` | Batch size for training | 8 |
| `LEARNING_RATE` | Learning rate | 2e-5 |
| `ADAPTER_SIZE` | Size of language adapters | 128 |
| `NUM_LABELS_SPAN` | Number of BIO tags | 3 |
| `NUM_LABELS_POLARITY` | Number of polarity classes | 4 |
| `NUM_LABELS_INTENSITY` | Number of intensity classes | 3 |

