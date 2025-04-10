# chatbot.py
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model
import random
import os

class ChatBot:
    def __init__(self, model_path='trained_model.h5', intents_path='intents.json', 
                 words_path='words.pkl', classes_path='classes.pkl'):
        # Load the trained model and necessary data
        self.lemmatizer = WordNetLemmatizer()
        self.model = load_model(model_path)
        self.intents = json.loads(open(intents_path).read())
        self.words = pickle.load(open(words_path, 'rb'))
        self.classes = pickle.load(open(classes_path, 'rb'))
    
    def clean_up_sentence(self, sentence):
        """Tokenize and lemmatize the sentence"""
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    
    def bow(self, sentence):
        """Convert a sentence to bag-of-words representation"""
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)
    
    def predict_class(self, sentence):
        """Predict the intent class for a given sentence"""
        bow = self.bow(sentence)
        res = self.model.predict(np.array([bow]))[0]
        
        # Filter out predictions below a threshold
        threshold = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > threshold]
        
        # Sort by probability
        results.sort(key=lambda x: x[1], reverse=True)
        
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        
        return return_list
    
    def get_response(self, intent_list):
        """Get a response based on predicted intent"""
        tag = intent_list[0]['intent']
        list_of_intents = self.intents['intents']
        
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        
        return result
    
    def chat(self):
        """Start an interactive chat session"""
        print("Bot is ready! (type 'quit' to exit)")
        while True:
            message = input("You: ")
            if message.lower() == "quit":
                break
            
            ints = self.predict_class(message)
            res = self.get_response(ints)
            print(f"Bot: {res}")
