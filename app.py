import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

#Monkey patch to avoid Streamlit's crash on torch.classes
import types
if not hasattr(torch, 'classes'):
    torch.classes = types.SimpleNamespace()

model_name = "hate-speech-distilroberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

id2label = {0: "hate", 1: "offensive", 2: "not hate"}
model.config.id2label = id2label

st.title("🛡️ Hate Speech Detector")
text = st.text_input("Enter text for hate detection:")

if text:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze()

    prediction = torch.argmax(probs).item()
    confidence = probs[prediction].item()

    st.markdown(f"**Prediction:** `{id2label[prediction]}`")
    st.markdown(f"**Confidence:** `{confidence:.2%}`")
