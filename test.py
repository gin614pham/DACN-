import streamlit as st
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import text
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the AWE model
@st.cache_resource(max_entries=1)
def load_awe_model():
    base_options = python.BaseOptions(model_asset_path="model.tflite")
    options = text.TextClassifierOptions(base_options=base_options)
    return text.TextClassifier.create_from_options(options)

# Load the BERT model
@st.cache_resource(max_entries=1)
def load_bert_model():
    base_options_bert = python.BaseOptions(model_asset_path="model_BERT.tflite")
    options_bert = text.TextClassifierOptions(base_options=base_options_bert)
    return text.TextClassifier.create_from_options(options_bert)

# Load the CNN model
@st.cache_resource(max_entries=1)
def load_cnn_model():
    model = load_model("sentiment_analysis_model.h5")
    tokenizer = Tokenizer()
    df = pd.read_csv('a1_IMDB_Dataset.csv')
    reviews = df['review'].tolist()
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)
    max_len = max([len(seq) for seq in sequences])
    return model, tokenizer, max_len

# Streamlit app
st.title('Text Classification App')

user_input = st.text_area("Enter your text here:")

# create selectbox for model selection
model_choice = st.selectbox('Select Model', ('AWE', 'CNN', 'BERT'))

if st.button('Classify') and user_input:
    if model_choice == 'AWE':
        awe_model = load_awe_model()
        classification_result = awe_model.classify(user_input)
        top_category = classification_result.classifications[0].categories[0]
        st.write(f'Input: {user_input}')
        category = 'Positive' if top_category.category_name == '1' else 'Negative'
        st.write(f'Category: {category}')
        st.write(f'Score: {top_category.score * 100:.2f}%')
    elif model_choice == 'CNN':
        loaded_model_cnn, cnn_tokenizer, max_len = load_cnn_model()
        new_reviews = [user_input]
        new_sequences = cnn_tokenizer.texts_to_sequences(new_reviews)
        new_X = pad_sequences(new_sequences, maxlen=max_len)
        predictions = loaded_model_cnn.predict(new_X)
        sentiment = "positive" if predictions > 0.5 else "negative"
        score = predictions[0] * 100
        st.write(f'Input: {user_input}')
        st.write(f"Predicted Sentiment: {sentiment}")
        st.write(f"Score: {score[0]:.2f}%")
    elif model_choice == 'BERT':
        bert_model = load_bert_model()
        classification_result = bert_model.classify(user_input)
        top_category = classification_result.classifications[0].categories[0]
        st.write(f'Input: {user_input}')
        category = 'Positive' if top_category.category_name == '1' else 'Negative'
        st.write(f'Category: {category}')
        st.write(f'Score: {top_category.score * 100:.2f}%')
else:
    st.write("Please enter some text to classify.")
