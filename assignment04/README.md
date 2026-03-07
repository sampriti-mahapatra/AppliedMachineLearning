# AML Assignment 4 — Containerization & CI

## Files

| File | Description |
|------|-------------|
| `app.py` | Flask web app with a `/score` POST endpoint for spam detection |
| `score.py` | Scoring function that uses the trained model and vectorizer |
| `model.pkl` | Trained RandomForest classifier |
| `vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Docker container recipe for the Flask app |
| `test.py` | 10 pytest tests (unit, integration, and Docker tests) |
| `coverage.txt` | Test coverage report (94% coverage) |
| `pre-commit` | Git pre-commit hook that runs tests before commits on main |

## Running the Docker Container

```bash
# Build the image
docker build -t aml4-flask-app .

# Run the container
docker run -d -p 5001:5001 --name aml4-container aml4-flask-app

# Test the endpoint
curl -X POST http://localhost:5001/score \
  -H "Content-Type: application/json" \
  -d '{"text": "Win free money now", "threshold": 0.5}'

# Stop and remove
docker stop aml4-container && docker rm aml4-container
```

## Running Tests

```bash
pip install -r requirements.txt
pytest test.py -v
```

## Setting Up the Pre-Commit Hook

After cloning the repo, activate the hook:

```bash
cp pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

Now every `git commit` on the `main` branch will automatically run `pytest test.py` first. If any test fails, the commit is blocked.
