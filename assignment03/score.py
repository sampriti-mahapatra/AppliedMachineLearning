import joblib
import os

# Load the vectorizer once at module level
_vectorizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vectorizer.pkl')
vectorizer = joblib.load(_vectorizer_path)


def score(text: str, model, threshold: float) -> tuple:
    """
    Score a text using a trained sklearn model.

    Args:
        text: Input text to classify.
        model: A trained sklearn estimator with predict_proba method.
        threshold: Decision threshold for classification.

    Returns:
        prediction (bool): True if spam (propensity >= threshold), False otherwise.
        propensity (float): Probability of the positive class (spam).
    """
    text_transformed = vectorizer.transform([text])
    propensity = float(model.predict_proba(text_transformed)[0][1])
    prediction = propensity >= threshold
    return prediction, propensity
