# 🎭 Sentiment Analyzer — ML Project

A binary sentiment classifier built with **TF-IDF + Logistic Regression**.

## Project Structure

```
sentiment_analyzer/
├── data.py                    # Dataset (200 labeled sentences, no downloads needed)
├── model.py                   # Pipeline, training, evaluation, plots
├── Sentiment_Analyzer.ipynb   # Guided walkthrough notebook
├── sentiment_model.pkl        # Saved model (created after running model.py)
└── plots/                     # Auto-generated evaluation charts
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── top_features.png
    └── score_distribution.png
```

## Quick Start

```bash
# 1. Train the model and generate plots
python model.py

# 2. Open the notebook
jupyter notebook Sentiment_Analyzer.ipynb
```

## Requirements

```
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install with:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

## Model Pipeline

```
Raw text  →  TF-IDF (1+2 grams, 5 000 features)  →  Logistic Regression
```

## Example Predictions

```python
from model import load_model, predict

model = load_model()
results = predict([
    "Absolutely loved it, highly recommend!",
    "Terrible quality, broke after one day.",
], model)
print(results)
```

## Results

| Metric   | Score  |
|----------|--------|
| Accuracy | ~0.71  |
| ROC-AUC  | ~0.79  |

## Ideas to Improve
- Use a larger public dataset (IMDb, SST-2, Yelp)
- Try `LinearSVC` or gradient-boosted trees
- Replace TF-IDF with word embeddings (GloVe, Word2Vec)
- Fine-tune a transformer model (BERT, DistilBERT)
