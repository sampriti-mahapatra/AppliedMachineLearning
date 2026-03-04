<html>
<body>

<h1>AML Assignment 3 — Unit Testing &amp; Flask Serving</h1>

<h2>The Task</h2>

<h3>1) Unit Testing</h3>
<p>In <code>score.py</code>, write a function with the following signature that scores a trained model on a text:</p>
<pre><code>def score(text:str,
                model:sklearn.estimator,
                threshold:float) -> prediction:bool,
                                                 propensity:float</code></pre>
<p>In <code>test.py</code>, write a unit test function <code>test_score(...)</code> to test the score function.</p>
<p>You may reload and use the best model saved during experiments in <code>train.ipynb</code> (in joblib/pkl format) for testing the score function.</p>
<p>You may consider the following points to construct your test cases:</p>
<ul>
  <li>does the function produce some output without crashing (smoke test)</li>
  <li>are the input/output formats/types as expected (format test)</li>
  <li>is prediction value 0 or 1 (sanity check)</li>
  <li>is propensity score between 0 and 1 (sanity check)</li>
  <li>if you put the threshold to 0 does the prediction always become 1 (edge case input)</li>
  <li>if you put the threshold to 1 does the prediction always become 0 (edge case input)</li>
  <li>on an obvious spam input text is the prediction 1 (typical input)</li>
  <li>on an obvious non-spam input text is the prediction 0 (typical input)</li>
</ul>

<h3>2) Flask Serving</h3>
<p>In <code>app.py</code>, create a flask endpoint <code>/score</code> that receives a text as a POST request and gives a response in the json format consisting of prediction and propensity.</p>
<p>In <code>test.py</code>, write an integration test function <code>test_flask(...)</code> that does the following:</p>
<ul>
  <li>launches the flask app using command line (e.g. use os.system)</li>
  <li>test the response from the localhost endpoint</li>
  <li>closes the flask app using command line</li>
</ul>
<p>In <code>coverage.txt</code> produce the coverage report output of the unit test and integration test using pytest.</p>

<h3>Reference Links</h3>
<ul>
  <li><a href="https://docs.pytest.org/en/8.0.x/">pytest documentation</a></li>
  <li><a href="https://pytest-cov.readthedocs.io/en/latest/reporting.html">pytest-cov reporting</a></li>
  <li><a href="https://flask.palletsprojects.com/en/2.3.x/quickstart/">Flask quickstart</a></li>
</ul>

<hr>

<h2>File Descriptions</h2>

<h3>Core Files</h3>
<table border="1" cellpadding="8" cellspacing="0">
  <tr><th>File</th><th>Description</th></tr>
  <tr>
    <td><code>score.py</code></td>
    <td>Contains the <code>score(text, model, threshold)</code> function that loads the vectorizer, transforms raw text into TF-IDF features, and returns a binary prediction and spam propensity score.</td>
  </tr>
  <tr>
    <td><code>test.py</code></td>
    <td>Contains 8 unit tests for the <code>score()</code> function (smoke, format, sanity, edge case, and typical input tests) and 1 integration test that launches the Flask app, sends HTTP requests, and verifies responses.</td>
  </tr>
  <tr>
    <td><code>app.py</code></td>
    <td>A Flask application exposing a POST <code>/score</code> endpoint on port 5001 that accepts JSON with a text field and returns a JSON response containing prediction and propensity.</td>
  </tr>
  <tr>
    <td><code>coverage.txt</code></td>
    <td>The pytest-cov coverage report output showing which lines of <code>score.py</code> and <code>app.py</code> were exercised by the tests.</td>
  </tr>
</table>

<h3>Extra Files</h3>
<table border="1" cellpadding="8" cellspacing="0">
  <tr><th>File</th><th>Description</th></tr>
  <tr>
    <td><code>prepare_artifacts.py</code></td>
    <td>A setup script that loads the trained RandomForestClassifier from AML_2, re-fits the TfidfVectorizer on the same training data, and saves both as <code>.pkl</code> files in this directory.</td>
  </tr>
  <tr>
    <td><code>model.pkl</code></td>
    <td>The serialized (joblib) RandomForestClassifier trained in AML_2. It takes numeric TF-IDF vectors as input and outputs class probabilities for ham vs spam.</td>
  </tr>
  <tr>
    <td><code>vectorizer.pkl</code></td>
    <td>The serialized (joblib) TfidfVectorizer fitted on the AML_2 training data. It converts raw text strings into the numeric TF-IDF feature vectors that the model expects.</td>
  </tr>
</table>

</body>
</html>
