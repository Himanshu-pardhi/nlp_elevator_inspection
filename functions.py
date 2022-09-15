import re
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import pickle
from textblob import TextBlob
from textblob.translate import NotTranslated
import random
sr = random.SystemRandom()
import numpy as np
from sklearn.preprocessing import LabelEncoder
import nlpaug.augmenter.word as naw

aug = naw.ContextualWordEmbsAug(model_path='bert-base-multilingual-uncased', aug_p=0.1)
model_embed = SentenceTransformer('all-MiniLM-L6-v2')
le = LabelEncoder()

# some text cleaning functions
def convert_to_lower(text):
    return text.lower()

def remove_numbers(text):
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
    return without_number

def remove_extra_white_spaces(text):
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc

def remove_special_char(text):
    special_char = r'[^\w\s]|.:,*"'
    remove_special_char = re.sub(pattern=special_char, repl=" ", string=text)
    return remove_special_char


def clean_message(message):
   message = convert_to_lower(message)
   message = remove_numbers(message)
   message = remove_extra_white_spaces(message)
   message = remove_special_char(message)
   return message

def embed_text(message):
    
    model_embed = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model_embed.encode(message)
    return embeddings


def predict_message(message):
    ma_cl = clean_message(message)
    em_me= embed_text(ma_cl)
    filename = 'C:\\Users\\user\\Desktop\\AI projects\\nlp_project_files\\LinearSVC_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict(em_me)
    return y_pred

def clean_data(df):
    """ Function to apply all in one
    parameters: dataframe
    return: dataframe  """

    # df['text'] = df['text'].apply(lambda x: convert_to_lower(x))
    df['text'] = df['text'].apply(lambda x: remove_numbers(x))
    df['text'] = df['text'].apply(lambda x: remove_extra_white_spaces(x))
    df['text'] = df['text'].apply(lambda x: remove_special_char(x))
    df = df.drop_duplicates()
    return df

def data_Aug(messege,aug_range=1):
    """ Function for augmenting data using Contextual Word Embeddings Augmenter (BERT)
    parameters: message: text from the dataset
                aug_range: required sampels number
                
    return : one augmented message   """

    augmented_messages = []
    for j in range(0,aug_range) :
        augmented_text = aug.augment(messege)
        augmented_messages.append(str(augmented_text))
        

    return augmented_messages

def data_augmentation(message, language, aug_range=1):
    augmented_messages = []
    if hasattr(message, "decode"):
        message = message.decode("utf-8")

    for j in range(0,aug_range) :
        new_message = ""
        text = TextBlob(message)
        try:
            text = text.translate(to=sr.choice(language))   ## Converting to random langauge for meaningful variation
            text = text.translate(to="en")
        except NotTranslated:
            pass
        augmented_messages.append(str(text))

    return augmented_messages

def embed(data, model_embed):
    sentences = data.values
    embeddings = model_embed.encode(sentences)
    return embeddings
from sklearn.preprocessing import LabelEncoder
# load the dataset
def load_dataset(df):
	# load the dataset as a numpy array
	data = df
	# retrieve numpy array
	data = df[['text', 'label']]
	# split into input and output elements
	X, y_label = data['text'], data['label']
	# label encode the target variable to have the classes 0 and 1
	y = le.fit_transform(y_label)
	return X, y_label,y   
def predict(file):
    """ Function to predict the classes for each entity using saved LinearSVC model
    parameter: json_file
    return: dataframe has the actual classes and the predicted classes """

    with open(file) as f:
        data = json.load(f)
        df_json=pd.DataFrame(data)
    df_json_training= df_json.loc[df_json['source']== 'TRAINING',:]      # choose the training source and drop the workflow
    df_json_training_fr = df_json_training.loc[df_json_training['culture']=='fr-fr',:]          # choose the French culture
    df_cleaned = clean_data(df_json_training_fr)   # data preprocessing
    print(df_cleaned)
    X, y_test_label, y_test = load_dataset(df_cleaned)           # feature selection
    new_X_test= embed(X,model_embed)              # implement the embeddings for the new test dataset 
    filename = 'C:\\Users\\user\\Desktop\\AI projects\\nlp_project_files\\LinearSVC_model.sav'
              
    loaded_model = pickle.load(open(filename, 'rb'))    # load the saved model
    y_pred = loaded_model.predict(new_X_test)           # predict the classes
    predicted_df = pd.DataFrame() 
    predicted_df['class']= y_test_label                      # create a data frame that show the prediction result
    predicted_df['y_test']= y_test
    predicted_df['y_pred']=y_pred
    
    return predicted_df
