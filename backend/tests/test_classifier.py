"""
Tests for the Classifier (unit tests with mocked model).
"""
import sys
from pathlib import Path
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def dummy_image(tmp_path):
    """Create a small dummy JPEG for testing."""
    from PIL import Image
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    p = tmp_path / "test_image.jpg"
    img.save(str(p))
    return str(p)


def test_preprocess_returns_correct_shape(dummy_image):
    from app.core.preprocessing import preprocess
    tensor = preprocess(dummy_image)
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_from_bytes():
    import io
    from PIL import Image
    from app.core.preprocessing import preprocess
    import numpy as np

    buf = io.BytesIO()
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(buf, format="JPEG")
    tensor = preprocess(buf.getvalue())
    assert tensor.shape == (1, 3, 224, 224)


def test_classifier_predict_output_shape(dummy_image):
    """Classifier.predict should return a dict containing label, confidence, all_probs, top_3, uncertainty_flags, is_ood, and entropy."""
    from app.core.classifier import Classifier

    clf = Classifier()

    # Patch model so we don't need real weights
    mock_model = MagicMock()
    mock_logits = torch.zeros(1, 2)
    mock_model.return_value = mock_logits

    with patch.object(Classifier, "_load_model"):
        clf.model = mock_model
        prediction = clf.predict(dummy_image)

    assert isinstance(prediction["label"], str)
    assert isinstance(prediction["confidence"], float)
    assert len(prediction["all_probs"]) == 2
    assert len(prediction["top_3"]) > 0
    assert abs(sum(prediction["all_probs"]) - 1.0) < 1e-5, "Probabilities should sum to 1"


def test_classifier_confidence_range(dummy_image):
    from app.core.classifier import Classifier

    clf = Classifier()
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(1, 2)

    with patch.object(Classifier, "_load_model"):
        clf.model = mock_model
        prediction = clf.predict(dummy_image)

    confidence = prediction["confidence"]
    all_probs = prediction["all_probs"]
    assert 0.0 <= confidence <= 1.0
    for p in all_probs:
        assert 0.0 <= p <= 1.0
