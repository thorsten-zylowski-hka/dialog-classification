from transformers import pipeline
from lime.lime_text import LimeTextExplainer
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ZeroShotLabelClassifier:

    def __init__(self, labels, model_name='facebook/bart-large-mnli', hypothesis_template='This example is {}.'):
        self.model_name = model_name
        self.classifier = pipeline('zero-shot-classification', model=model_name, hypothesis_template=hypothesis_template) 
        self.labels = labels


    def classify(self, text):
        result = self.classifier(text, self.labels)
        return result


class ZeroShotLabelClassifierExplainer:
    
    def __init__(self, model_name='facebook/bart-large-mnli', sep_token='</>', hypothesis_template='This example is {}.'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.sep_token = sep_token
        self.explainer = LimeTextExplainer(class_names=['contradiction', 'neutral', 'entailment'])
        self.hypothesis_template = hypothesis_template

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
    
    def normalize_array(self, arr, min, max):
        return [(x - min) / (max - min) for x in arr]
    
    def get_normalized_explanations(self, explanation_list):
        positive_keys = []
        positive_values = []
        negative_keys = []
        negative_values = []
        for h in explanation_list:   
            key = h['key']
            value = h['value']
            if value >=0:            
                positive_keys.append(key)
                positive_values.append(value)
            else:
                negative_keys.append(key)
                negative_values.append(-value)

        minimum = min(positive_values + negative_values)
        maximum = max(positive_values + negative_values)

        normalized_positive_values = self.normalize_array(positive_values, min=minimum, max=maximum)
        normalized_negative_values = self.normalize_array(negative_values, min=minimum, max=maximum)

        normalized_positive_highlights = []
        for i in range(len(normalized_positive_values)):
            normalized_positive_highlights.append({
                        "key":positive_keys[i],
                        "value": normalized_positive_values[i]
                    }) 
            
        normalized_negative_highlights = []
        for i in range(len(normalized_negative_values)):
            normalized_negative_highlights.append({
                        "key":negative_keys[i],
                        "value": normalized_negative_values[i]
                    }) 
            
        return {
            "positive": normalized_positive_highlights,
            "negative": normalized_negative_highlights
        } 
    
    def get_user_centered_explanation(self, label, explanation_list):
        return {
            "global_model_description": "Foo bar",
            "high_level_highlight_explanation": "Die Wörter Foo und Bar haben den größten Einfluss auf das Attribut label.",
            "counterfactuals": [
                {
                    "counterfactual": "I don't like drawing on canvas.",
                    "classification": {
                        'labels': [ 'artistic',
                                    'enterprising',
                                    'social',
                                    'conventional',
                                    'realistic',
                                    'investigative'],
                        'scores': [ 0.8289536237716675,
                                    0.07523660361766815,
                                    0.0406937301158905,
                                    0.020833542570471764,
                                    0.01776549592614174,
                                    0.01651705801486969]
                    }
                }
            ]
        }
        

    def explain(self, text, label, num_features=15, num_samples=100, normalize=False, user_centered_explanation = True):
        
        explanation_result = {
            "label": label            
        }
        
        self.hypothesis_template.replace('{}', label)+'.'
        exp_text = text + '</>' + self.hypothesis_template.replace('{}', label)
        expanded_num_features = num_features + len(self.hypothesis_template.split())
        explanation = self.explainer.explain_instance(exp_text, self.predictor, num_features=expanded_num_features, num_samples=num_samples, top_labels=3)

        explanation_list = explanation.as_list(label=2)
        explanation_list = [{"key": item[0], "value": item[1] } for item in explanation_list if not (item[0] in self.hypothesis_template.replace('{}', label) and item[0] not in text)]        
        
        explanation_result['highlights'] = explanation_list

        if normalize:
            normalized_highlights = self.get_normalized_explanations(explanation_list)
            explanation_result['normalized_highlights'] = normalized_highlights
        
        explanation_result['is_normalized'] = normalize

        if user_centered_explanation:
            user_centered_explanation_result = self.get_user_centered_explanation(label, explanation_list)
            explanation_result['user_centered_explanation'] = user_centered_explanation_result

        return explanation_result