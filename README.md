# Zero-shot Label Classifier and Explainer

Zero-shot label classification based on Huggingface's zero-shot classification pipeline with the ability to explain the classification using the LIME framework. For a simple classification in English all default values can be used. The model "facebook/bart-large-mnli" is used for this. No further hypothesis template (see below) is needed either.

For the classification, the text to be classified as well as the possible labels have to be specified. 

For the explanation, the text and the label to be explained are required. In the example, the explanation is chosen for the label "artistic", which had received the highest value during classification.

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

## More languages
By the choice of the model, the classification as well as the explanation can be used for different languages. On the one hand, the model to be used is passed via the model_name parameter. On the other hand, a hypothesis template must be specified in the corresponding language, which corresponds to the English default hypothesis template "Thist example is {}.". The curly brackets are mandatory, since at this place the appropriate label is inserted.

The following example shows the usage for the German language.

```python
riasec = ['konventionell', 'realistisch', 'investigativ', 'unternehmerisch', 'sozial', 'künstlerisch']
text = 'Ich male gerne mit Ölfarben auf Leinwände.'

model_name = 'svalabs/gbert-large-zeroshot-nli'
hypothesis_template = "Dieser Satz ist {}." 

label_classifier = ZeroShotLabelClassifier(labels=riasec, model_name=model_name, hypothesis_template=hypothesis_template)
classification_result = label_classifier.classify(text)

print("Classification Result:", classification_result)

explainer = ZeroShotLabelClassifierExplainer(model_name=model_name, hypothesis_template=hypothesis_template)
most_likely_label = classification_result['labels'][0]
explanation = explainer.explain(text, label=most_likely_label)

print("Explanation:", explanation)
```