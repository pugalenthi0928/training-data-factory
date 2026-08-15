"""Tests for difficulty calibration module."""

from __future__ import annotations

from training_data_robo.difficulty import calibrate_batch, calibrate_difficulty


class TestCalibrateDifficulty:
    def test_easy_example(self):
        ex = {
            "id": "1",
            "input_text": "What color is the sky?",
            "output_text": "The sky is blue.",
        }
        result = calibrate_difficulty(ex)
        assert result["difficulty"] == "easy"
        assert "difficulty_features" in result
        assert result["difficulty_features"]["raw_score"] < 2.0

    def test_hard_example(self):
        ex = {
            "id": "2",
            "input_text": "Compare and contrast supervised and unsupervised learning. Analyze the trade-offs and explain why one might be preferred.",
            "output_text": (
                "Step 1: First, let us define supervised learning. It is a paradigm where "
                "models learn from labeled data. Therefore, the algorithm can map inputs to outputs. "
                "Step 2: Second, unsupervised learning uses unlabeled data. Consequently, it discovers "
                "hidden patterns. Furthermore, it requires different evaluation strategies. "
                "Step 3: The implications of choosing one over the other depend on data availability. "
                "In conclusion, supervised learning is preferred when labeled data is abundant, "
                "whereas unsupervised approaches are advantageous for exploratory analysis. "
                "Nevertheless, the trade-off between annotation cost and performance must be "
                "carefully evaluated. The relationship between these paradigms is complementary."
            ),
        }
        result = calibrate_difficulty(ex)
        assert result["difficulty"] == "hard"
        assert result["difficulty_features"]["reasoning_signals"] >= 3
        assert result["difficulty_features"]["complexity_signals"] >= 2

    def test_medium_example(self):
        ex = {
            "id": "3",
            "input_text": "How does photosynthesis work?",
            "output_text": (
                "Photosynthesis is the process by which plants convert sunlight into energy. "
                "First, chlorophyll in the leaves absorbs light energy from the sun. "
                "This energy is then used to convert carbon dioxide and water into glucose and oxygen. "
                "Therefore, photosynthesis is essential for life on Earth."
            ),
        }
        result = calibrate_difficulty(ex)
        assert result["difficulty"] in ("medium", "easy")  # borderline

    def test_preserves_existing_fields(self):
        ex = {"id": "x", "input_text": "Q", "output_text": "A", "custom_field": 42}
        calibrate_difficulty(ex)
        assert ex["custom_field"] == 42


class TestCalibrateBatch:
    def test_batch(self):
        examples = [
            {"id": "1", "input_text": "Q", "output_text": "Short."},
            {
                "id": "2",
                "input_text": "Compare X",
                "output_text": "A" * 600 + " therefore because furthermore step 1 step 2",
            },
        ]
        summary = calibrate_batch(examples)
        assert summary["total"] == 2
        assert "distribution" in summary
        assert sum(summary["distribution"].values()) == 2

    def test_empty_batch(self):
        summary = calibrate_batch([])
        assert summary["total"] == 0
