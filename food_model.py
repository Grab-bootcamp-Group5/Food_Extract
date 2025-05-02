from typing import List, Union
import numpy as np
import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
from food_extractor.data_utils import id2tag, id2tag_no_prod

HF_MODEL_PATH = "chambliss/distilbert-for-food-extraction"

class FoodModel:
    
    def __init__(self, model_path: str = HF_MODEL_PATH, no_product_labels: bool = False):
        if model_path == HF_MODEL_PATH: self.model = DistilBertForTokenClassification.from_pretrained(HF_MODEL_PATH)
        else: self.model = DistilBertForTokenClassification.from_pretrained(model_path)
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased", do_lower_case=False)
        
        self.label_dict = id2tag if no_product_labels == False else id2tag_no_prod

        self.model.to(self.device)
        self.model.eval()


    def ids_to_labels(self, label_ids: list) -> list:
        return [self.label_dict[tensor.item()] for tensor in label_ids]


    def predict(self, texts: Union[str, List[str]], entities_only: bool = False):
        if type(texts) == str: texts = [texts]

        n_examples = len(texts)
        encodings = self.tokenizer(texts, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt")
        encodings.to(self.device)

        logits = self.model.forward(encodings["input_ids"])[0]
        probs_per_token = torch.nn.functional.softmax(logits, dim=2)
        max_probs_per_token = torch.max(probs_per_token, dim=2)
        probs, preds = max_probs_per_token.values, max_probs_per_token.indices
        labels = [self.ids_to_labels(p) for p in preds]

        pred_summaries = [self.create_pred_summary(encodings[i], labels[i], probs[i]) for i in range(n_examples)]
        entities = [self.process_pred(pred_summary, text) for pred_summary, text in zip(pred_summaries, texts)]
        if entities_only: return entities

        for pred_summary, ent_dict in zip(pred_summaries, entities):
            pred_summary["entities"] = ent_dict
            
        return pred_summaries


    def extract_foods(self, text: Union[str, List[str]]) -> dict:
        if type(text) == str: text = [text]
        batch_entities = self.predict(text, entities_only=True)

        for entities in batch_entities:
            for ent_type in entities:
                ents = entities[ent_type]
                for ent in ents:
                    if len(ent["text"]) <= 3: entities[ent_type].remove(ent)
                    if ent_type == "Product":
                        if ent["text"][0].islower():
                            if ent in ents: entities[ent_type].remove(ent)
                            
        return batch_entities


    def create_pred_summary(self, encoding, labels, probs):
        mask = encoding.attention_mask
        tokens = mask_list(encoding.tokens, mask)
        labels = mask_list(labels, mask)
        offsets = mask_list(encoding.offsets, mask)
        prob_list = mask_list(probs.tolist(), mask)
        pred_summary = {
            "tokens": tokens,
            "labels": labels,
            "offsets": offsets,
            "probabilities": prob_list,
            "avg_probability": np.mean(prob_list),
            "lowest_probability": np.min(prob_list),
        }
        return pred_summary


    def process_pred(self, pred_summary: dict, orig_str: str) -> dict:
        labels = pred_summary["labels"]
        offsets = pred_summary["offsets"]
        probs = pred_summary["probabilities"]

        entities = {"Product": [], "Ingredient": []}
        entity_start, entity_end = None, None
        entity_start_idx, entity_end_idx = None, None

        for i, label in enumerate(labels):
            if label == "O": continue

            prev_prefix, prev_label_type, next_prefix, next_label_type = get_prev_and_next_labels(i, labels)
            prefix, label_type = label.split("-")

            if prefix == "B":
                if next_label_type != label_type or next_prefix != "I":
                    start, end = offsets[i]
                    entity = orig_str[start:end]
                    entities[label_type].append({"text": entity, "span": [start, end], "conf": probs[i]})
                else:  
                    entity_start = offsets[i][0]
                    entity_start_idx = i
                    continue

            if prefix == "I":
                if i == 0 or prev_prefix == "O" or (prev_prefix in ["B", "I"] and label_type != prev_label_type):
                    entity_start = offsets[i][0]
                    entity_start_idx = i

                if next_label_type != label_type or next_prefix != "I":
                    entity_end = offsets[i][1]
                    entity_end_idx = i
                    entity = orig_str[entity_start:entity_end]
                    entities[label_type].append({"text": entity,
                                                "span": [entity_start, entity_end],
                                                "conf": np.mean(probs[entity_start_idx : entity_end_idx + 1])})
                else: continue

        return entities


def mask_list(orig_list: list, mask: list) -> list:
    masked = [item for item, m in zip(orig_list, mask) if m == 1]
    return masked


def get_prev_and_next_labels(idx: int, labels: List[str]):
    is_first = idx == 0
    is_last  = idx == (len(labels) - 1)

    if is_first:
        prev_label, prev_prefix, prev_label_type = "O", None, None
    else:
        prev_label = labels[idx - 1]
        if prev_label != "O":
            prev_prefix, prev_label_type = prev_label.split("-")
        else:
            prev_prefix, prev_label_type = "O", None
            
    if is_last:
        next_label, next_prefix, next_label_type = "O", None, None
    else:
        next_label = labels[idx + 1]
        if next_label != "O":
            next_prefix, next_label_type = next_label.split("-")
        else:
            next_prefix, next_label_type = "O", None

    return prev_prefix, prev_label_type, next_prefix, next_label_type


examples = """3 tablespoons (21 grams) blanched almond flour
... ¾ teaspoon pumpkin spice blend
... ⅛ teaspoon baking soda
... ⅛ teaspoon Diamond Crystal kosher salt
... 1½ tablespoons maple syrup or 1 tablespoon honey
... 1 tablespoon (15 grams) canned pumpkin puree
... 1 teaspoon avocado oil or melted coconut oil
... ⅛ teaspoon vanilla extract
... 1 large egg""".split("\n")

model = FoodModel()
model.extract_foods(examples)