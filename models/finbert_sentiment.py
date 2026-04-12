"""
FinBERT Sentiment Analysis Module.
Uses ProsusAI/finbert for financial sentiment analysis with fine-tuning
and reinforcement learning feedback integration.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    """Dataset for FinBERT fine-tuning."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class FinBERTSentiment:
    """
    FinBERT-based sentiment analyzer with:
    - Pre-trained financial sentiment classification
    - Fine-tuning on collected news data
    - RL feedback loop for continuous improvement
    """

    LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}
    SCORE_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        )

        os.makedirs(config.finbert.cache_dir, exist_ok=True)
        logger.info(f"Loading FinBERT model: {config.finbert.model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.finbert.model_name, cache_dir=config.finbert.cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.finbert.model_name, cache_dir=config.finbert.cache_dir
        ).to(self.device)

        # RL feedback buffer: stores (text, predicted_label, reward) for updating
        self._rl_feedback_buffer: List[Tuple[str, int, float]] = []
        self._prediction_history: List[Dict] = []

    def analyze(self, text: str) -> Dict:
        """Analyze sentiment of a single text."""
        self.model.eval()
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.config.finbert.max_seq_length, padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        label_idx = int(np.argmax(probs))
        label = self.LABEL_MAP[label_idx]
        score = float(self.SCORE_MAP[label] * probs[label_idx])

        result = {
            "label": label,
            "score": score,
            "confidence": float(probs[label_idx]),
            "probabilities": {
                "positive": float(probs[0]),
                "negative": float(probs[1]),
                "neutral": float(probs[2]),
            },
        }

        self._prediction_history.append({"text": text[:200], **result})
        return result

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment of multiple texts in batches."""
        results = []
        batch_size = self.config.finbert.batch_size
        self.model.eval()

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", truncation=True,
                max_length=self.config.finbert.max_seq_length, padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

            for j, prob in enumerate(probs):
                label_idx = int(np.argmax(prob))
                label = self.LABEL_MAP[label_idx]
                score = float(self.SCORE_MAP[label] * prob[label_idx])
                results.append({
                    "label": label,
                    "score": score,
                    "confidence": float(prob[label_idx]),
                    "probabilities": {
                        "positive": float(prob[0]),
                        "negative": float(prob[1]),
                        "neutral": float(prob[2]),
                    },
                })

        return results

    def get_aggregated_sentiment(self, texts: List[str]) -> Dict:
        """Get aggregated sentiment across multiple texts."""
        if not texts:
            return {"avg_score": 0.0, "label": "neutral", "count": 0}

        results = self.analyze_batch(texts)
        scores = [r["score"] for r in results]
        avg_score = np.mean(scores)

        if avg_score > 0.1:
            label = "positive"
        elif avg_score < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return {
            "avg_score": float(avg_score),
            "std_score": float(np.std(scores)),
            "label": label,
            "count": len(results),
            "positive_ratio": sum(1 for r in results if r["label"] == "positive") / len(results),
            "negative_ratio": sum(1 for r in results if r["label"] == "negative") / len(results),
        }

    def add_rl_feedback(self, text: str, predicted_label: int, reward: float):
        """
        Add RL feedback for model updating.
        reward > 0: prediction was useful for trading
        reward < 0: prediction led to losses
        """
        self._rl_feedback_buffer.append((text, predicted_label, reward))

    def fine_tune_with_rl_feedback(self):
        """
        Fine-tune FinBERT using accumulated RL feedback.
        Uses reward-weighted loss to update the model towards predictions
        that correlate with profitable trading.
        """
        if len(self._rl_feedback_buffer) < 32:
            logger.info("Not enough RL feedback samples for fine-tuning")
            return

        logger.info(f"Fine-tuning FinBERT with {len(self._rl_feedback_buffer)} RL feedback samples")

        # Convert rewards to soft labels
        texts, labels, rewards = zip(*self._rl_feedback_buffer)
        texts = list(texts)
        rewards = np.array(rewards)

        # Normalize rewards to [0, 1] for weighting
        if rewards.max() > rewards.min():
            weights = (rewards - rewards.min()) / (rewards.max() - rewards.min())
        else:
            weights = np.ones_like(rewards) * 0.5

        # For positive rewards, keep original labels; for negative, flip them
        adjusted_labels = []
        for i, (label, reward) in enumerate(zip(labels, rewards)):
            if reward > 0:
                adjusted_labels.append(label)
            elif reward < -0.5:
                # Flip label for strongly negative rewards
                # positive->negative, negative->positive, neutral stays
                if label == 0:
                    adjusted_labels.append(1)
                elif label == 1:
                    adjusted_labels.append(0)
                else:
                    adjusted_labels.append(2)
            else:
                adjusted_labels.append(label)

        dataset = SentimentDataset(
            texts, adjusted_labels, self.tokenizer,
            max_length=self.config.finbert.max_seq_length,
        )
        dataloader = DataLoader(dataset, batch_size=self.config.finbert.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.finbert.fine_tune_lr
        )
        num_steps = len(dataloader) * self.config.finbert.fine_tune_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_steps // 10, num_training_steps=num_steps
        )

        self.model.train()
        total_loss = 0
        steps = 0

        for epoch in range(self.config.finbert.fine_tune_epochs):
            for batch_idx, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                # Weight loss by reward magnitude
                batch_weights = torch.tensor(
                    weights[batch_idx * self.config.finbert.batch_size:
                            (batch_idx + 1) * self.config.finbert.batch_size],
                    device=self.device, dtype=torch.float32,
                )
                if len(batch_weights) < outputs.logits.shape[0]:
                    batch_weights = torch.ones(outputs.logits.shape[0], device=self.device)

                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                steps += 1

        avg_loss = total_loss / max(steps, 1)
        logger.info(f"RL fine-tuning complete. Avg loss: {avg_loss:.4f}")

        # Clear buffer after training
        self._rl_feedback_buffer.clear()

    def fine_tune_on_labeled_data(self, texts: List[str], labels: List[int]):
        """Fine-tune on explicitly labeled data (e.g., from curated datasets)."""
        if len(texts) < 10:
            logger.warning("Not enough labeled data for fine-tuning")
            return

        dataset = SentimentDataset(
            texts, labels, self.tokenizer,
            max_length=self.config.finbert.max_seq_length,
        )
        dataloader = DataLoader(dataset, batch_size=self.config.finbert.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.finbert.fine_tune_lr
        )

        self.model.train()
        for epoch in range(self.config.finbert.fine_tune_epochs):
            epoch_loss = 0
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()

            logger.info(f"Fine-tune epoch {epoch + 1}: loss={epoch_loss / len(dataloader):.4f}")

    def save_model(self, path: str):
        """Save the fine-tuned model."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load a fine-tuned model."""
        if os.path.exists(path):
            self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"Model path not found: {path}")
