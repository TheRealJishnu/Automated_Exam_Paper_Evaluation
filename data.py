import nltk
from nltk.corpus import stopwords
import re
from striprtf.striprtf import rtf_to_text
from gensim.models.doc2vec import TaggedDocument
import os

tagged_data = []
doc = []
directory = "Dataset/Q4"

for filename in os.listdir(directory):
    if filename.endswith(".rtf"):
        with open(os.path.join(directory, filename), "rb") as file:
            rtf_text = file.read().decode("utf-8")
            # Convert .rtf to plain text
            text = rtf_to_text(rtf_text)
            text = re.sub(r'\[[0-9]*\]',' ',text)
            text = re.sub(r'\s+',' ',text)
            text = re.sub(r'\d+',' ',text)
            text = re.sub(r'\s+',' ',text)
            text = text.lower()
            tokens = nltk.word_tokenize(text)
            words = [word for word in tokens if word.isalpha()]
            words = [word for word in words if word not in stopwords.words("english")]
            cleaned_text = " ".join(words)
            doc.append(cleaned_text)
            # Append a TaggedDocument to the list
            tagged_data.append(TaggedDocument(words=cleaned_text.split(), tags=[str(len(tagged_data))]))
