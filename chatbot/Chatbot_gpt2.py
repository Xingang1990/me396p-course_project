#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:26:36 2022

@author: xingangli
"""

# Setup
## for speech-to-text
import speech_recognition as sr

## for text-to-speech
from gtts import gTTS

## for playsound
from playsound import playsound

## for language model
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import os
from pathlib import Path
import datetime

# Build the AI
class ChatBot():
    def __init__(self, name):
        print("--- starting up AI Chatbot", name, "---")
        self.name = name

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            print("listening...")
            audio = recognizer.listen(mic)
        try:
            self.text = recognizer.recognize_google(audio)
            print("me --> ", self.text)
        except:
            print("me -->  ERROR")
            
        return self.text
    
    def take_user_input(self):
        user_input = input("Enter you question: ")
        self.text = user_input
        return user_input
                 
    @staticmethod
    def text_to_speech(text):
        current_dir = Path(__file__).resolve().parent
        print("Groot AI --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        response_time = datetime.datetime.now().time().strftime('%H%M%S')
        audio_dir = current_dir / f"response_{response_time}.mp3"
        speaker.save(audio_dir)
        playsound(f"response_{response_time}.mp3")
        # os.system(f"start {audio_dir}") #windows
        # os.system(f"rhythmbox-client --play {audio_dir}") #linux
        # os.system(f"afplay {audio_dir}") #mac
        # os.remove(audio_dir)

    def wake_up(self, text):
        return True if self.name in text.lower() else False

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')
    
#Randomly(from the top N probability distribution) select the next word
def choose_from_top(probs, n=3, random_seed=None):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    np.random.seed(random_seed)
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id) 
        
# Run the AI
if __name__ == "__main__":
    
    ai = ChatBot(name="groot")
    
    #load gpt2 
    #use CPU or GPU
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        
    #build the model structure
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model = model.to(device)
    
    #load the fine-tuned gpt2
    MODEL_EPOCH = 4
    current_dir = Path(__file__).resolve().parent
    models_dir = current_dir / "trained_models"
    trained_model_path = os.path.join(models_dir, f"gpt2_medium_pythonlecturer_{MODEL_EPOCH}.pt")
    model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # os.environ["TOKENIZERS_PARALLELISM"] = "true"

    shut_down = ["close yourself", "shut down", "go away", "turn off"]
    on = True

    while on:
        #take user input through typing
        user_input = ai.take_user_input()
        
        # #take user input through voice
        # user_input = ai.speech_to_text()

        ## wake up
        if ai.wake_up(ai.text) is True:
            res = "Hello I am Groot the AI, what can I do for you?"
            
        elif ai.text[:-1].lower() in "how are you doing?":
            res = np.random.choice(["I am good.", "I am fine.", "Pretty good."]) + " How are you?"
        
        # action time
        elif "time" in ai.text:
            res = ai.action_time()
        
        ## respond politely
        elif any(i in ai.text[:-1].lower() for i in ["thank","thanks","thank you"]):
            res = np.random.choice(["you're welcome!","anytime!","no problem!","cool!","I'm here if you need me!","peace out!"])
        
        elif any(i in ai.text for i in shut_down):
            res = np.random.choice(["See you.", "Bye!", "See you next time.", "See you later.", "Byebye"])
            on = False
            
        ##use the fine-tuned gpt2 to generate answers
        else:
            question = f"Q: {user_input.strip()} \n\n A:"
            with torch.no_grad():
            # for paragraph_idx in range(generated_paragraph):
                paragraph_finished = False
                cur_ids = torch.tensor(tokenizer.encode(question)).unsqueeze(0).to(device)
                
                max_paragraph_length = 150
                for i in range(max_paragraph_length):
                    outputs = model(cur_ids, labels=cur_ids)
                    loss, logits = outputs[:2]
                    softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
                    if i < 3:
                        n = 10 #num of top N probability distributions
                    else:
                        n = 5
                        
                    #setting random seed 
                    rand_seed = None #default: None, change this to a value for debugging purpose 
                    next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n, random_seed=rand_seed) 
                    cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

                    if next_token_id in tokenizer.encode('<|endoftext|>'):
                        paragraph_finished = True
                        break

                if paragraph_finished:
                    output_list = list(cur_ids.squeeze().to('cpu').numpy())
                    res = tokenizer.decode(output_list)[:-13] #not include '<|endoftext|>'
                    res = res[res.find("A:")+2:].strip()
                    # print(res)
                  
        ai.text_to_speech(res)



