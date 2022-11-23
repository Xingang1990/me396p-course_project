# Setup
## for speech-to-text
import speech_recognition as sr

## for text-to-speech
from gtts import gTTS

## for language model
import transformers
#get transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel


## for data
import os
import datetime
import numpy as np
from pathlib import Path
            
# Build the AI
class ChatBot():
    def __init__(self, name):
        print("--- starting up", name, "---")
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
    
    def take_user_input(self):
        self.text = input("Enter you question: ")
                 
    @staticmethod
    def text_to_speech(text):
        current_dir = Path(__file__).resolve().parent
        audio_dir = current_dir / "lecture.mp3"
        print("ai --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        speaker.save(audio_dir)
        os.system(f"rhythmbox-client --play {audio_dir}")  #mac->afplay | windows->start | linux->mgp123(installation) or rhythmbox
        # os.remove("res.mp3")

    def wake_up(self, text):
        return True if self.name in text.lower() else False

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')
    
   
# Run the AI
if __name__ == "__main__":
    
    ai = ChatBot(name="groot")
    nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium") #will download the model for the first use
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    shut_down = ["close yourself", "shut down", "go away", "turn off"]
    on_off = True

    while on_off:
        # ai.speech_to_text()
        user_input = ai.take_user_input()

        ## wake up
        if ai.wake_up(ai.text) is True:
            res = "Hello I am Groot the AI, what can I do for you?"
        
        # action time
        elif "time" in ai.text:
            res = ai.action_time()
        
        ## respond politely
        elif any(i in ai.text for i in ["thank","thanks"]):
            res = np.random.choice(["you're welcome!","anytime!","no problem!","cool!","I'm here if you need me!","peace out!"])
        
        elif any(i in ai.text for i in shut_down):
            res = np.random.choice(["See you.", "Bye!"])
            on_off = False
            
        ## conversation
        else:
            chat = nlp(transformers.Conversation(ai.text), pad_token_id=50256) #use the NLP model
            res = str(chat)
            res = res[res.find("bot >> ")+6:].strip()
        
        ai.text_to_speech(res)

