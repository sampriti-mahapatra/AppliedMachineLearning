import joblib
import os
import subprocess
import time
import pytest
import requests

from score import score
import app as flask_app

# Load the model once for all tests
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
model = joblib.load(MODEL_PATH)


# --- Unit Tests for score() ---

def test_score_smoke():
    """Smoke test: function runs without crashing."""
    prediction, propensity = score("hello world", model, 0.5)


def test_score_output_types():
    """Format test: output types are as expected."""
    prediction, propensity = score("hello world", model, 0.5)
    assert isinstance(prediction, bool), f"prediction should be bool, got {type(prediction)}"
    assert isinstance(propensity, float), f"propensity should be float, got {type(propensity)}"


def test_score_prediction_binary():
    """Sanity check: prediction is True or False (0 or 1)."""
    prediction, _ = score("test message", model, 0.5)
    assert prediction in (True, False)


def test_score_propensity_range():
    """Sanity check: propensity is between 0 and 1."""
    _, propensity = score("test message", model, 0.5)
    assert 0.0 <= propensity <= 1.0, f"propensity {propensity} not in [0, 1]"


def test_score_threshold_zero():
    """Edge case: threshold=0 should always predict True (1)."""
    texts = [
        "hello how are you",
        "Free money click now!!!",
        "meeting at 3pm tomorrow",
    ]
    for text in texts:
        prediction, _ = score(text, model, threshold=0.0)
        assert prediction is True, f"threshold=0 should predict True for: {text}"


def test_score_threshold_one():
    """Edge case: threshold=1 should always predict False (0)."""
    texts = [
        "hello how are you",
        "Free money click now!!!",
        "meeting at 3pm tomorrow",
    ]
    for text in texts:
        prediction, _ = score(text, model, threshold=1.0)
        assert prediction is False, f"threshold=1 should predict False for: {text}"


def test_score_obvious_spam():
    """Typical input: obvious spam text should be predicted as spam (1)."""
    spam_text = "WINNER!! You have been selected for a free cash prize! Call 09061234567 NOW to claim your reward. Txt STOP to cancel."
    prediction, propensity = score(spam_text, model, 0.5)
    assert prediction is True, f"obvious spam should be predicted as True, got propensity={propensity}"


def test_score_obvious_ham():
    """Typical input: obvious non-spam text should be predicted as not spam (0)."""
    ham_text = "Hey, are we still meeting for lunch tomorrow at noon?"
    prediction, propensity = score(ham_text, model, 0.5)
    assert prediction is False, f"obvious ham should be predicted as False, got propensity={propensity}"


# --- Docker Test ---

DOCKER_IMAGE = "aml4-flask-app"
DOCKER_CONTAINER = "aml4-test-container"


def test_docker():
    """Docker test: build image, run container, test /score endpoint, clean up."""
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the Docker image
    build_result = os.system(f"docker build -t {DOCKER_IMAGE} {project_dir}")
    assert build_result == 0, "Docker build failed"

    # Run the container
    os.system(f"docker rm -f {DOCKER_CONTAINER} 2>/dev/null")
    run_result = os.system(
        f"docker run -d -p 5001:5001 --name {DOCKER_CONTAINER} {DOCKER_IMAGE}"
    )
    assert run_result == 0, "Docker run failed"

    try:
        # Wait for the container to be ready
        time.sleep(5)

        # Send a request to the /score endpoint
        response = requests.post(
            "http://localhost:5001/score",
            json={"text": "WINNER! Free cash prize! Call now to claim!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "propensity" in data
        assert data["prediction"] in (0, 1)
        assert isinstance(data["propensity"], float)
        assert 0.0 <= data["propensity"] <= 1.0

    finally:
        # Stop and remove the container
        os.system(f"docker stop {DOCKER_CONTAINER}")
        os.system(f"docker rm {DOCKER_CONTAINER}")


# --- Integration Test for Flask app ---

def test_flask():
    """Integration test: launch Flask app, test /score endpoint, shut down."""
    # Launch Flask app as a subprocess
    proc = subprocess.Popen(
        ['python', 'app.py'],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for the server to start
        time.sleep(3)

        # Test with a spam message
        response = requests.post(
            'http://127.0.0.1:5001/score',
            json={'text': 'WINNER! Free cash prize! Call now to claim!'}
        )
        assert response.status_code == 200
        data = response.json()
        assert 'prediction' in data
        assert 'propensity' in data
        assert data['prediction'] in (0, 1)
        assert 0.0 <= data['propensity'] <= 1.0

        # Test with a ham message
        response = requests.post(
            'http://127.0.0.1:5001/score',
            json={'text': 'Hey, are we meeting for lunch tomorrow?'}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['prediction'] in (0, 1)
        assert 0.0 <= data['propensity'] <= 1.0

    finally:
        # Shut down the Flask app
        proc.terminate()
        proc.wait(timeout=5)
