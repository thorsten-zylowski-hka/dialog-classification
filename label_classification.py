from transformers import pipeline
from lime.lime_text import LimeTextExplainer
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ZeroShotLabelClassifier:

    def __init__(self, labels, model_name='facebook/bart-large-mnli'):
        self.model_name = model_name
        self.classifier = pipeline('zero-shot-classification', model=model_name) 
        self.labels = labels
        self.explainer = LimeTextExplainer(class_names=['contradiction', 'neutral', 'entailment'])


    def classify(self, text):
        result = self.classifier(text, self.labels)
        return result


class ZeroShotLabelClassifierExplainer:
    
    def __init__(self, model_name='facebook/bart-large-mnli', sep_token='</>'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.sep_token = sep_token
        self.explainer = LimeTextExplainer(class_names=['contradiction', 'neutral', 'entailment'])

    def predictor(self, texts):
        inputs = []
        for text in texts:
            text_parts = text.split(self.sep_token)
            premise = text_parts[0]
            hypothesis = text_parts[1]
            inputs.append((premise, hypothesis))
        outputs = self.model(**self.tokenizer(inputs, return_tensors="pt", padding=True))
        tensor_logits = outputs[0]
        probas = F.softmax(tensor_logits, dim=1).detach().numpy()    
        return probas

    def explain(self, text, label, num_features=15, num_samples=100):
        exp_text = text + '</>' + 'This example is ' + label + '.'
        explanation = self.explainer.explain_instance(exp_text, self.predictor, num_features=num_features, num_samples=num_samples, top_labels=3)
        return explanation.as_list(label=2)
