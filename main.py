import transformers
import torch

torch.cuda.empty_cache()
import gc

gc.collect()
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from torch.autograd import Function
from transformers import AutoTokenizer
from utils import *
import get_data


from flask import Flask, jsonify, render_template, request
rep_pretrained_model_name_sentiment = "HooshvareLab/bert-fa-base-uncased-sentiment-deepsentipers-binary"
rep_pretrained_model_name_ner = "HooshvareLab/bert-base-parsbert-ner-uncased"

pretrained_tokenizer_sentiment = AutoTokenizer.from_pretrained(rep_pretrained_model_name_sentiment)
pretrained_tokenizer_ner = AutoTokenizer.from_pretrained(rep_pretrained_model_name_ner)

sentiment_model = torch.load("./data/my_module_1400_02_10.pt", map_location='cpu')
sentiment_model.eval()
ner_model = torch.load("./data/NER_model.pt", map_location='cpu')
ner_model.eval()
with open('./data/peyma_ner_tag2id.pickle', 'rb') as handle:
    peyma_ner_tag2id = pickle.load(handle)
with open('./data/peyma_ner_id2tag.pickle', 'rb') as handle:
    peyma_ner_id2tag = pickle.load(handle)

sentiment_tag2id = {"positive": 1, "negative": 0}
sentiment_id2tag = {1: "positive", 0: "negative"}

app = Flask(__name__)

form_html = """<br />
                <form method="get">
                    Text:
                    <br />
                    <textarea name="text" dir="rtl" style="width:1000" rows="5" maxlength="400">{text}</textarea>
<br /><br />
                <input type="submit" name="submit" value="Get">

                </form>
<br />======================== <br />
{result}

"""


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


@app.route('/sentiment/', methods=['GET'])
def sentiment():
    try:
        text = request.args.get('text')

        data = pretrained_tokenizer_sentiment(text, padding="max_length", truncation=True, return_tensors="pt", max_length=32)
        output = sentiment_model(input_ids=data["input_ids"], attention_mask=data["token_type_ids"],
                                 token_type_ids=data["attention_mask"])
        sentiment_output = output["logits_similarity"].cpu().detach().numpy()
        sentiment_output_softmax = softmax(sentiment_output[0])
        label = np.argmax(sentiment_output_softmax)
        confidence = np.max(sentiment_output_softmax)
        result = "Label:\t" + str(sentiment_id2tag[label]) + "<br /> Confidences:\t" + str(sentiment_output_softmax) + "<br /> Logits:\t" + str(sentiment_output)
    except Exception as e: 
        text = ""
        result = str(e)
    return form_html.format_map({"text": text, "result": result})


@app.route('/ner/', methods=['GET'])
def ner():
    try:

        text = request.args.get('text')

        data = pretrained_tokenizer_ner(text, return_tensors="pt")
        output = ner_model(input_ids=data["input_ids"], attention_mask=data["token_type_ids"],
                           token_type_ids=data["attention_mask"])

        logits_ner = output["logits_ner"].cpu().detach().numpy()
        logits_ner = np.concatenate(logits_ner, axis=0)
        logits_ner = np.argmax(logits_ner, axis=-1)
        logits_ner = logits_ner.tolist()
        words = data["input_ids"][0].cpu().detach().tolist()
        tags = [peyma_ner_id2tag[i] for i in logits_ner]
        
        result = str(tags) +"<br /><br /><br />"
        last_word = []
        last_tag = None
        for i in range(len(words)):
            tag = tags[i]
            if tag != "X" and not tag.startswith("I-"):
                if last_tag is not None and len(last_word)>0:
                    result += pretrained_tokenizer_ner.decode(last_word) + "\t---\t" + last_tag + "<br />"
                last_word = []
                last_tag = None

            if tag.startswith("B-") :
                last_tag = tag.replace("B-", "").replace("I-", "")
                last_word.append(words[i])

            if tag == "X" or tag.startswith("I-"):
                last_word.append(words[i])
        if last_tag is not None and len(last_word) > 0:
            result += pretrained_tokenizer.decode(last_word) + "\t---\t" + last_tag + "<br />"

    except Exception as e: 
        text = ""
        result = str(e)
    return form_html.format_map({"text": text, "result": result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9093, debug=False)
