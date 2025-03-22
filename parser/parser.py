import spacy
from spacy.tokens import Span
from spacy.language import Language
import PyPDF2
from spacy.util import filter_spans
import requests

def load_dut_spec(pdf_path):
    """Loads text from a DUT specification PDF."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

nlp = spacy.load("en_core_web_sm")

# Function to fetch entity labels dynamically from an LLM
def fetch_entity_labels():
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"
    headers = {"Authorization": "Bearer YOUR_HUGGINGFACE_API_KEY"}  # Replace with your API key
    data = {"inputs": "Extract key entity labels from DUT specifications, including signals, constraints, and operations."}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()[0]['generated_text'].strip()
    else:
        return {}

ENTITY_LABELS = fetch_entity_labels()  # Fetch labels dynamically

@Language.component("custom_entity_recognizer")
def custom_entity_recognizer(doc):
    """Custom Named Entity Recognition (NER) for DUT specifications"""
    custom_entities = []
    for label, keywords in ENTITY_LABELS.items():
        for keyword in keywords:
            start = 0
            while True:
                found_index = doc.text.find(keyword, start)
                if found_index == -1:
                    break
                end = found_index + len(keyword)
                span = doc.char_span(found_index, end, label=label)
                if span is not None:
                    custom_entities.append(span)
                start = end
    doc.ents = filter_spans(list(doc.ents) + custom_entities)
    return doc

nlp.add_pipe("custom_entity_recognizer", after="ner")

def parse_dut_text(text):
    """Parses DUT specification text and extracts relevant entities."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
