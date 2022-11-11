# Zero-shot Label Classifier and Explainer

## Usage
```python
from label_classification import ZeroShotLabelClassifier
from label_classification import ZeroShotLabelClassifierExplainer

riasec = ['conventional', 'realistic', 'investigative', 'enterprising', 'social', 'artistic']
text = 'I like to draw paintings with oil on a canvas.'

label_classifier = ZeroShotLabelClassifier(labels=riasec)
classification_result = label_classifier.classify(text)

print("Classification Result:", classification_result)

explainer = ZeroShotLabelClassifierExplainer()
most_likely_label = classification_result['labels'][0]
explanation = explainer.explain(text, label=most_likely_label)

print("Explanation:", explanation)
```