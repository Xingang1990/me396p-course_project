# ME396p Course Project
This repo is created for codes and files of the course project of ME396p.

## Setup

### Training/finetuning GPT2
1. Training data preparation: See all the data files in the folder `training_data_preparation`. We manually obtained the scripts of the python lectures as TXT files on Youtube from the links as listed below. Some scripts have periods for separating sentences (`period.txt`) while some do not (`no-period.txt`). For these with periods, we treat each sentence (segmented by period) as one data point. For these with NO periods, we decided to concatenate `n` (we used `n=5` in our project) lines of texts as one data point. It should be noted that one full sentence might be segmented in this way. We provid a Python program to parse the two TXT files to two CSV files (`data.csv` and `data2.csv`) for the training purpose. The two CSV files are combined to `trainingdata.csv` for finetuning the GPT2 model.
2. We finetune the GPT2 model on Kaggle and here is the [link](https://www.kaggle.com/code/xingangli/gpt2-finetuning) for the codes. You can edit the codes as you want and interact with codes like a Jupyter notebook. Note: Remember to download the finetuned models (e.g., `gpt2_medium_pythonlecturer_4.pt`) to your local computer otherwise they will be erased after a period of no interaction.
3. The finetuned GPT2 model is then used as the AI backend for the chatbot. We provide a [finetuned GPT2 model](https://drive.google.com/drive/folders/1jJQTTJPZVU_nmTskp9HxT5CY01bgbR6K?usp=share_link) here. Download and put it in the folder `trained_models` under the folder `chatbot`.

### AI chatbot
1. Conda environment (example): ```conda create -n chatbot python=3.10.6 ```
2. Install requirements:
    - `pytorch` tested with 1.13.0, gpu not required
    - `transformers` tested with 4.23.1
    - `SpeechRecognition` tested with 3.8.1
    - `gTTs` tested with 2.2.4
    - `PyAudio` tested with 0.2.12
    - `playsound` tested with 1.2.2
3. Start the AI chatbot: cd to `chabot` and run the chatbot ```python chatbot_gpt2.py```. It will download the original GPT2 and load its neural network architecture, which might take a few minutes on your first use. Downloading GPT2 will not be needed in the future. Note: The AI chatbot can take either typing texts or voice as input by commenting one of the two lines of codes: ```user_input = ai.take_user_input()``` and ```user_input = ai.speech_to_text()```.
4. Enjoy your conversation with the AI chatbot talking about Python by inputing questions, like "How to debug in python?", "What are the common built-in data types in python?".

## Resourses for the project

### AI chatbot
- https://towardsdatascience.com/ai-chatbot-with-nlp-speech-recognition-transformers-583716a299e9
- https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313
- https://medium.com/ai-innovation/beginners-guide-to-retrain-gpt-2-117m-to-generate-custom-text-content-8bb5363d8b7f 

### GPT2 model finetuing
- https://huggingface.co/blog/how-to-generate
- https://gist.github.com/mf1024/3df214d2f17f3dcc56450ddf0d5a4cd7

### GPT2 and Transformers detailed explanation
- http://jalammar.github.io/illustrated-gpt2/
- https://www.youtube.com/watch?v=MQnJZuBGmSQ
- http://jalammar.github.io/illustrated-transformer/
- https://www.youtube.com/watch?v=MQnJZuBGmSQ

### Youtube videos used for obtaining the scripts of python lectures
- https://www.youtube.com/watch?v=_uQrJ0TkZlc&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=1&t=18275s
- https://www.youtube.com/watch?v=rfscVS0vtbw&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=2 
- https://www.youtube.com/watch?v=XKHEtdqhLK8&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=3 
- https://www.youtube.com/watch?v=8DvywoWv6fI&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=4 
- https://www.youtube.com/watch?v=eWRfhZUzrAc&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=5 
- https://www.youtube.com/watch?v=t8pPdKYpowI&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=6 
- https://www.youtube.com/watch?v=HGOBQPFzWKo&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=7 
- https://www.youtube.com/watch?v=WGJJIrtnfpk&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=8 
- https://www.youtube.com/watch?v=B9nFMZIYQl0&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=9 
- https://www.youtube.com/watch?v=jH85McHenvw&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=10 
- https://www.youtube.com/watch?v=kWEbNBXc2-Y&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=11 
- https://www.youtube.com/watch?v=LzYNWme1W6Q&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=12 
- https://www.youtube.com/watch?v=sxTmJE4k0ho&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=13 
- https://www.youtube.com/watch?v=T936yTchDck&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=14 
- https://www.youtube.com/watch?v=mJEpimi_tFo&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=15 
- https://www.youtube.com/watch?v=pJ3IPRqiD2M&list=PL926BsUgkCkIETA9WMlTKjgKg2XZN8AF6&index=16 

## Lighting Talk
### Our slides
- https://colab.research.google.com/drive/11nqgPYmgU0984CIP7ep1JaSWvK2fYFYx?usp=sharing 
