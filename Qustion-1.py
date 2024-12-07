from transformers import pipeline
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')
nltk.download('wordnet')

def get_antonym(word):
    """Find the antonym of a given word using WordNet."""
    antonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    return antonyms[0] if antonyms else word

def reverse_sentiment(sentence, sentiment):
    """Reverse the sentiment of a sentence."""
    words = sentence.split()
    if sentiment == "POSITIVE":
        return " ".join([get_antonym(word) for word in words])
    elif sentiment == "NEGATIVE":
        return " ".join([get_antonym(word) for word in words])
    return sentence  

def process_text(input_text):
    sentiment_model = pipeline("sentiment-analysis")

    sentences = sent_tokenize(input_text)
    reversed_sentences = []

    for sentence in sentences:
        result = sentiment_model(sentence)[0]
        sentiment = result['label']
        reversed_sentence = reverse_sentiment(sentence, sentiment)
        reversed_sentences.append(reversed_sentence)

    return " ".join(reversed_sentences)

input_text = """The product is amazing and works perfectly. I had an issue with the delivery, but customer service was helpful. Overall, I am satisfied with my purchase."""

output_text = process_text(input_text)
print(output_text)
