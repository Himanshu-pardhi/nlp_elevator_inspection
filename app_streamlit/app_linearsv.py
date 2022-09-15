import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
import pickle
import re
from sklearn import metrics
import json

le = LabelEncoder()
model_embed = SentenceTransformer('all-MiniLM-L6-v2')
model=pickle.load(open('/home/soukaina/Downloads/LinearSVC_model.sav','rb'))

def main():
    """Classifier"""
    st.title("Classification")
    html_temp = """
    <div style="background-color
    :blue;padding:10px">
    <h1 style="color:white;text-align:center;">Invoice Classification ML App </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    activity = ['Prediction']
    choice = st.sidebar.selectbox("Select Activity", activity)

    if choice == 'Prediction':
        st.info("Prediction with ML")
        ml_model = ["LinearSVC model"]
        model_choice = st.selectbox("Select Model", ml_model,key = "<uniquevalueofsomesort>")



        @st.cache
        def remove_numbers(text):
            number_pattern = r'\d+'
            without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
            return without_number

        @st.cache
        def remove_extra_white_spaces(text):
            single_char_pattern = r'\s+[a-zA-Z]\s+'
            without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
            return without_sc

        @st.cache
        def remove_special_char(text):
            special_char = r'[^\w\s]|.:,*"'
            remove_special_char = re.sub(pattern=special_char, repl=" ", string=text)
            return remove_special_char

        @st.cache
        def clean_data(df):
            """ Function to apply all in one
            parameters: dataframe
            return: dataframe  """

            df['text'] = df['text'].apply(lambda x: remove_numbers(x))
            df['text'] = df['text'].apply(lambda x: remove_extra_white_spaces(x))
            df['text'] = df['text'].apply(lambda x: remove_special_char(x))
            df = df.drop_duplicates()
            return df

        @st.cache
        def embed(df, model_embed):
            sentences = df.values
            embeddings = model_embed.encode(sentences)
            return embeddings

        @st.cache
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
            return X, y_label, y


        uploaded_file = st.file_uploader("Please upload the json file here")
        if uploaded_file is not None:
            df_json = pd.read_json(uploaded_file)
            st.write(df_json)

        if st.button("Predict"):
            # df_json_training = df_json.loc[df_json['source'] == 'TRAINING',:]  # choose the training source and drop the workflow

            df_json_training_fr = df_json.loc[df_json['culture'] == 'fr-fr',:]  # choose the French culture
            df_cleaned = clean_data(df_json_training_fr)  # data preprocessing
            # X, y_test = load_dataset(df_cleaned)           # feature selection
            X, y_test_label, y_test = load_dataset(df_cleaned)
            new_X_test = embed(X, model_embed)  # implement the embeddings for the new test dataset
            model_path = '/home/soukaina/Downloads/LinearSVC_model.sav'

            loaded_model = pickle.load(open(model_path, 'rb'))  # load the saved model
            y_pred = loaded_model.predict(new_X_test)  # predict the classes
            predicted_df = pd.DataFrame()  # create a data frame that show the prediction result
            predicted_df['class'] = y_test_label
            predicted_df['y_test'] = y_test
            predicted_df['y_pred'] = y_pred
            st.write(predicted_df)
            st.write(metrics.classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()