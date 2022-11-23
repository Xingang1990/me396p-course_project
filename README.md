# ME396p Course Project
This repo is created for codes and files of the course project of ME396p.


## Setup

### AI chatbot
1. Conda environment (example): ```conda create -n chatbot python=3.10.6 ```
2. Install requirements:
    - `pytorch` tested with 1.13.0, gpu not required
    - `transformers` tested with 4.23.1
    - `SpeechRecognition` tested with 3.8.1
    - `gTTs` tested with 2.2.4
    - `PyAudio` tested with 0.2.12


### Training of GPT2
1. Training data preparation: See all the data files in the folder `training_data_preparation`. We manually obtained the scripts of the python lectures as TXT files on Youtube from the links as listed below. Some scripts have periods for separating sentences (`period.txt`) while some do not (`no-period.txt`). For these with periods, we treat each sentence (segmented by period) as one data point. For these with NO periods, we decided to concatenate `n` (we used `n=5` in our project) lines of texts as one data point. It should be noted that one sentence might be segmented in this way. We provided a Python program to parse the two TXT files to two CSV files (`data.csv` and `data2.csv`) for the training purpose. The two CSV files are combined to `trainingdata.csv` for finetuning the GPT2 model.
2. We finetune the GPT2 model on Kaggle and here is the link for the codes. https://www.kaggle.com/code/xingangli/gpt2-finetuning
3. The finetuned GPT2 model is then used as the AI backend for the chatbot.

## Resourses for the project

### AI chatbot
- https://towardsdatascience.com/ai-chatbot-with-nlp-speech-recognition-transformers-583716a299e9

- https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

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
- 
## Lighting Talk
### Our slides
- https://colab.research.google.com/drive/11nqgPYmgU0984CIP7ep1JaSWvK2fYFYx?usp=sharing 
