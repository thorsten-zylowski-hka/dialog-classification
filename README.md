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

## Return values
Classifier and explainer both return dictionaries which can easily be transformed to JSON.

### Classification
```json
{
    "sequence": "Ich male gerne mit Ölfarben auf Leinwände.",
    "labels": [
        "künstlerisch",
        "sozial",
        "investigativ",
        "konventionell",
        "unternehmerisch",
        "realistisch"
    ],
    "scores": [
        0.6761776804924011,
        0.12924496829509735,
        0.06201709061861038,
        0.0512198880314827,
        0.04103587940335274,
        0.04030449688434601
    ]
}
```

### Explanation
```json
{
    "highlights": [
        {
            "key": "gerne",
            "value": 0.02039188985575313
        },
        {
            "key": "Ich",
            "value": -0.014657435456174138
        },
        {
            "key": "Leinwände",
            "value": 0.006532542314393964
        },
        {
            "key": "male",
            "value": 0.002843916273899499
        },
        {
            "key": "Ölfarben",
            "value": 0.002825456106864587
        },
        {
            "key": "auf",
            "value": 0.002699091365190593
        },
        {
            "key": "mit",
            "value": -0.0012255975250213022
        }
    ]
}
```
## More languages
By the choice of the model, the classification as well as the explanation can be used for different languages. On the one hand, the model to be used is passed via the model_name parameter. On the other hand, a hypothesis template must be specified in the corresponding language, which corresponds to the English default hypothesis template "This example is {}.". The curly brackets are mandatory, since at this place the appropriate label is inserted.

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