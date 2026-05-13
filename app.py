import gradio as gr
import joblib
import re

# Załadowanie modeli z tego samego folderu
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

polish_stopwords = {
    'i', 'w', 'na', 'z', 'do', 'o', 'ze', 'za', 'po', 'od', 'się', 'jest', 'są', 
    'nie', 'tak', 'ale', 'a', 'to', 'jak', 'że', 'co', 'ten', 'ta', 'oraz', 'lub', 
    'czy', 'dla', 'tym', 'tam', 'tu', 'niż', 'był', 'była', 'było', 'były', 'jego'
}

def clean_polish_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-ząćęłńóśźż\s]', '', text)
    words = text.split()
    cleaned_words = [word for word in words if word not in polish_stopwords and len(word) > 1]
    return " ".join(cleaned_words)

def analyze(text):
    clean_text = clean_polish_text(text)
    text_tfidf = vectorizer.transform([clean_text])
    prob = model.predict_proba(text_tfidf)
    
    prob_neg, prob_pos = prob[0][0], prob[0][1]
    
    if prob_pos > 0.5:
        label = "Pozytywny"
        confidence = prob_pos
    else:
        label = "Negatywny"
        confidence = prob_neg
        
    return f"Status recenzji: **{label}** (Pewność modelu: {confidence:.1%})"

with gr.Blocks() as demo:
    gr.Markdown("# 🇵🇱 Analizator Sentymentu (Lokalny)")
    text_input = gr.Textbox(lines=3, placeholder="Wpisz recenzję po polsku...")
    btn = gr.Button("Analizuj sentyment", variant="primary")
    output = gr.Markdown()
    btn.click(analyze, inputs=text_input, outputs=output)

if __name__ == "__main__":
    demo.launch()


