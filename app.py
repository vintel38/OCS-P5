import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import joblib
import tensorflow_hub as hub 
model_USE = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

import gradio as gr

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
tokenizer = nltk.RegexpTokenizer(r'(?:C\+\+)|(?:c\+\+)|(?:c\#)|(?:C\#)|(?:\.net)|(?:\.NET)|\w{2,}')

label_bin, USE_clf = joblib.load('USE_clf.pkl')

def preprocess(text):
	
    # cleaning, tokenization 
    text = text.lower()
    text_lst = tokenizer.tokenize(text)
    texty = []
    for wrd in text_lst:
        if not wrd in stop_words and not wrd.isnumeric():
            texty.append(stemmer.stem(wrd))
    return texty
    
def make_predictions(text):

    # preprocessing
    token_lst = preprocess(text)
    
    # USE enconding
    embeddings = model_USE([' '.join(token_lst)]).numpy()
    
    # classifier predictions 
    pred_USE_clf = USE_clf.predict(embeddings)
    
    # convert back to tags words
    tags = label_bin.inverse_transform(pred_USE_clf)
    
    return tags
    
output = gr.Interface(
    fn=make_predictions,
    inputs=[
        gr.inputs.Textbox(lines=5, placeholder="Enter text here...")],
    outputs=[gr.outputs.Textbox(label="Predicted Tags")]
)

output.launch()