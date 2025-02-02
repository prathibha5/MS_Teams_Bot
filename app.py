from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import json

app = Flask(__name__)
CORS(app)

# Load the classification model and tokenizer
model_dir = './expense_report_model'
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load the label mapping
with open('./label_mapping.json', 'r') as f:
    label_mapping = json.load(f)
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Load translation model
translation_model_name = "facebook/m2m100_418M"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)

# Language code mapping
LANG_CODE_MAPPING = {
    'zh-cn': 'zh',
    'zh-tw': 'zh',
    'tw': 'zh',
    'en-gb': 'en',
    'es-la': 'es',
    'fr-ca': 'fr',
    'pt-br': 'pt'
}

def translate_to_english(text, source_lang):
    mapped_lang = LANG_CODE_MAPPING.get(source_lang, source_lang)
    translation_tokenizer.src_lang = mapped_lang
    inputs = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = translation_model.generate(
            **inputs,
            forced_bos_token_id=translation_tokenizer.get_lang_id("en")
        )

    return translation_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

def classify_text(input_text, source_lang):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if source_lang != 'en':
        input_text = translate_to_english(input_text, source_lang)

    model.eval()
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = F.softmax(logits, dim=1)
    predicted_label_idx = probabilities.argmax(dim=1).item()
    predicted_label = reverse_label_mapping[predicted_label_idx]

    label_probabilities = {reverse_label_mapping[i]: prob.item() for i, prob in enumerate(probabilities[0])}

    return predicted_label, label_probabilities

@app.route('/classify', methods=['POST'])
def classify():
    # Check if the request is JSON or form data
    if request.is_json:
        data = request.json
    else:
        data = request.form

    query = data.get('query', '')
    source_lang = data.get('source_lang', 'en')

    try:
        predicted_label, probabilities = classify_text(query, source_lang)
        return jsonify({
            'predicted_label': predicted_label,
            'probabilities': probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return '''
    <html>
        <head>
            <title>Text Classification API</title>
        </head>
        <body>
            <h1>Text Classification API</h1>
            <form action="/classify" method="post">
                <label for="query">Enter text:</label><br>
                <textarea id="query" name="query" rows="4" cols="50"></textarea><br>
                <label for="source_lang">Source Language:</label><br>
                <input type="text" id="source_lang" name="source_lang" value="en"><br><br>
                <input type="submit" value="Classify">
            </form>
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)



