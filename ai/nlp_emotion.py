"""
EmotiCam NLP Text Emotion Recognition & Multimodal Fusion Engine.
Analyzes textual sentiment/emotions based on standard NLP emotion categories
and fuses textual emotional signals with facial computer vision telemetry.
"""

import re
import math
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class TextEmotionAnalyzer:
    """
    Lightweight, deterministic, high-throughput NLP emotion analysis engine
    calibrated with semantic emotion lexicons matching standard NLP datasets (Joy, Sadness, Anger, Fear, Surprise, Love, Neutral).
    """

    # Emotion classes aligned with vision canonical classes + NLP love/joy mappings
    EMOTION_MAPPING = {
        "joy": "Happy",
        "love": "Happy",
        "sadness": "Sad",
        "anger": "Angry",
        "fear": "Fear",
        "surprise": "Surprise",
        "disgust": "Disgust",
        "neutral": "Neutral"
    }

    CANONICAL_CLASSES = ["Happy", "Neutral", "Surprise", "Sad", "Angry", "Disgust", "Fear"]

    # Comprehensive curated emotion lexicon dictionary with valence weights
    LEXICON = {
        "joy": {
            "happy", "delighted", "thrilled", "joyful", "excited", "ecstatic", "glad",
            "blessed", "cheerful", "content", "wonderful", "amazing", "awesome", "fantastic",
            "great", "superb", "brilliant", "enjoy", "pleased", "gleeful", "smiling",
            "celebrate", "radiant", "jubilant", "elated", "positive", "love", "loving",
            "adored", "beloved", "caring", "grateful", "appreciate", "sweet", "paradise"
        },
        "sadness": {
            "sad", "depressed", "sorrow", "unhappy", "gloomy", "heartbroken", "crying",
            "tears", "down", "disappointed", "miserable", "hurt", "grief", "mourning",
            "despair", "melancholy", "hopeless", "lonely", "alone", "regret", "pity",
            "tragic", "pain", "broken", "empty", "loss", "devastated", "gloomy", "sigh"
        },
        "anger": {
            "angry", "mad", "furious", "outraged", "irritated", "annoyed", "pissed",
            "hate", "enraged", "bitter", "resentful", "hostile", "frustrated", "rage",
            "infuriated", "displeased", "temper", "aggravated", "fuming", "agitated",
            "violent", "scorn", "disdain", "disgusted", "offensive", "cheat", "liar"
        },
        "fear": {
            "fear", "scared", "afraid", "terrified", "frightened", "anxious", "nervous",
            "panic", "horrified", "dread", "alarmed", "apprehensive", "worried", "shivering",
            "trembling", "creepy", "spooky", "danger", "threat", "paralyzed", "stress",
            "shocked", "intimidated", "uneasy"
        },
        "surprise": {
            "surprise", "surprised", "astonished", "amazed", "stunned", "shocked",
            "unexpected", "unbelievable", "wonder", "astounded", "jaw-dropping", "speechless",
            "abrupt", "startled", "unusual", "staggering", "whoa", "wow", "unreal"
        },
        "disgust": {
            "disgust", "disgusted", "revolted", "nauseated", "gross", "repulsed",
            "sickening", "vile", "foul", "awful", "nasty", "appalling", "loathsome",
            "repugnant", "distasteful", "yuck", "eww"
        }
    }

    INTENSIFIERS = {
        "very": 1.5, "extremely": 1.8, "super": 1.6, "really": 1.4, "so": 1.3,
        "deeply": 1.5, "totally": 1.4, "completely": 1.5, "utterly": 1.7, "absolutely": 1.6
    }

    NEGATIONS = {"not", "never", "no", "hardly", "barely", "scarcely", "without", "rarely"}

    def __init__(self):
        pass

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyzes the emotional polarity and distribution of the input string.
        Returns:
            - dominant_emotion: str (e.g. "Happy")
            - confidence: float (0.0 - 100.0)
            - distribution: Dict[str, float]
            - tokens_analyzed: int
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "dominant_emotion": "Neutral",
                "confidence": 50.0,
                "distribution": {e: (100.0 / len(self.CANONICAL_CLASSES)) for e in self.CANONICAL_CLASSES},
                "tokens_analyzed": 0,
                "text": ""
            }

        # Tokenize and normalize words
        words = re.findall(r'\b[a-zA-Z\']+\b', text.lower())
        if not words:
            return {
                "dominant_emotion": "Neutral",
                "confidence": 50.0,
                "distribution": {e: (100.0 / len(self.CANONICAL_CLASSES)) for e in self.CANONICAL_CLASSES},
                "tokens_analyzed": 0,
                "text": text
            }

        scores = {c: 0.1 for c in self.CANONICAL_CLASSES}
        scores["Neutral"] = 0.5  # Prior baseline

        # Scan text tokens with context window
        n = len(words)
        for i, word in enumerate(words):
            # Check context modifiers (intensifiers and negations in previous 2 words)
            multiplier = 1.0
            is_negated = False

            if i > 0:
                prev1 = words[i - 1]
                if prev1 in self.INTENSIFIERS:
                    multiplier *= self.INTENSIFIERS[prev1]
                if prev1 in self.NEGATIONS:
                    is_negated = True
            if i > 1:
                prev2 = words[i - 2]
                if prev2 in self.NEGATIONS:
                    is_negated = True

            matched = False
            for raw_cat, words_set in self.LEXICON.items():
                if word in words_set:
                    matched = True
                    canonical_cat = self.EMOTION_MAPPING.get(raw_cat, "Neutral")
                    weight = 1.5 * multiplier

                    if is_negated:
                        # Flip emotion polarity: positive -> neutral/sad, negative -> neutral
                        if canonical_cat == "Happy":
                            scores["Sad"] += weight * 0.8
                            scores["Neutral"] += weight * 0.4
                        else:
                            scores["Neutral"] += weight * 0.8
                    else:
                        scores[canonical_cat] += weight

        # Softmax normalization
        exp_scores = {k: math.exp(v) for k, v in scores.items()}
        total_exp = sum(exp_scores.values()) + 1e-9

        distribution = {
            k: round((exp_scores[k] / total_exp) * 100.0, 1)
            for k in self.CANONICAL_CLASSES
        }

        # Determine dominant emotion
        dominant_emotion = max(distribution, key=distribution.get)
        confidence = distribution[dominant_emotion]

        return {
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
            "distribution": distribution,
            "tokens_analyzed": len(words),
            "text": text
        }

    def fuse_multimodal(
        self,
        face_emotion: Dict[str, Any],
        text_emotion: Dict[str, Any],
        face_weight: float = 0.60
    ) -> Dict[str, Any]:
        """
        Performs weighted multimodal Bayesian-style late fusion of facial vision signals
        and NLP textual signals into a unified emotional telemetry state.
        """
        text_weight = 1.0 - face_weight
        combined_dist = {}

        face_dist = face_emotion.get("distribution", {})
        text_dist = text_emotion.get("distribution", {})

        for c in self.CANONICAL_CLASSES:
            f_val = face_dist.get(c, 0.0)
            t_val = text_dist.get(c, 0.0)
            combined_dist[c] = round((f_val * face_weight) + (t_val * text_weight), 1)

        # Normalize to sum 100%
        total = sum(combined_dist.values()) + 1e-9
        normalized_dist = {
            k: round((v / total) * 100.0, 1) for k, v in combined_dist.items()
        }

        dominant = max(normalized_dist, key=normalized_dist.get)
        conf = normalized_dist[dominant]

        # Determine emotional alignment / congruency
        face_dom = face_emotion.get("primary_emotion", face_emotion.get("dominant_emotion", "Neutral"))
        text_dom = text_emotion.get("dominant_emotion", "Neutral")
        is_congruent = (face_dom == text_dom)

        return {
            "multimodal_dominant_emotion": dominant,
            "multimodal_confidence": conf,
            "distribution": normalized_dist,
            "face_dominant_emotion": face_dom,
            "text_dominant_emotion": text_dom,
            "is_congruent": is_congruent,
            "fusion_weights": {
                "vision": round(face_weight, 2),
                "nlp": round(text_weight, 2)
            }
        }
