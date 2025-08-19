from flask import Flask, request, render_template
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from model_def import SimpleCNN
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

# Load models
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
bert_model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
model = SimpleCNN(kernel_size=3)
model.load_state_dict(torch.load("cnn_model_k3.pth", map_location="cpu"))
model.eval()

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    # Cleaning
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Case folding
    case_folded = cleaned_text.lower()
    
    # Tokenization
    tokens = case_folded.split()
    
    # Stemming
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    return {
        'original': text,
        'cleaned': cleaned_text,
        'case_folded': case_folded,
        'tokens': tokens,
        'stemmed_tokens': stemmed_tokens,
        'stemmed_text': ' '.join(stemmed_tokens)
    }

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state

def classify_text(text):
    embedding = get_bert_embedding(text)
    output = model(embedding)
    probs = F.softmax(output, dim=1)
    pred_class = torch.argmax(probs).item()
    label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
    
    processed_embedding = torch.mean(embedding, dim=1).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    token_embeddings = embedding.squeeze(0).tolist()
    
    return label_map[pred_class], probs.squeeze().tolist(), processed_embedding, tokens, token_embeddings

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form["text"]
        if not text.strip():
            raise ValueError("Text tidak boleh kosong")
        
        # Preprocessing
        preprocessing_steps = preprocess_text(text)
        
        # Classification
        label, probs, embedding, tokens, token_embeddings = classify_text(preprocessing_steps['stemmed_text'])
        
        return render_template(
            "index.html",
            input_text=text,
            prediction=label,
            probabilities=probs,
            embedding=embedding,
            tokens=tokens,
            token_embeddings=token_embeddings,
            preprocessing=preprocessing_steps,
            zip=zip
        )
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)