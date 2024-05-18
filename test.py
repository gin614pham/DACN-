import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import text
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf

base_options = python.BaseOptions(model_asset_path="model.tflite")
options = text.TextClassifierOptions(base_options=base_options)
classifier = text.TextClassifier.create_from_options(options)

base_options_bert = python.BaseOptions(model_asset_path="model_BERT.tflite")
options_bert = text.TextClassifierOptions(base_options=base_options_bert)
classifier_bert = text.TextClassifier.create_from_options(options_bert)

loaded_model = load_model('my_model.h5')

# Tạo ứng dụng Streamlit
st.title('Text Classification App')

user_input = st.text_area("Enter your text here:")

# create selectbox for model selection
model_choice = st.selectbox('Select Model', ('AWE', 'CNN', 'BERT'))

if st.button('Classify'):
    if user_input:
        # switch model
        if model_choice == 'AWE':
            classification_result = classifier.classify(user_input)
            top_category = classification_result.classifications[0].categories[0]
            st.write(f'Input: {user_input}')
            # if category_name = 1 then Negative else Positive
            category = 'Positive' if top_category.category_name == '0' else 'Negative'
            st.write(f'Category: {category}')
            st.write(f'Score: {top_category.score * 100:.2f}%')
        elif model_choice == 'CNN':

            pass
        elif model_choice == 'BERT':
            classification_result = classifier_bert.classify(user_input)
            top_category = classification_result.classifications[0].categories[0]
            st.write(f'Input: {user_input}')
            # if category_name = 1 then Negative else Positive
            category = 'Positive' if top_category.category_name == '0' else 'Negative'
            st.write(f'Category: {category}')
            st.write(f'Score: {top_category.score * 100:.2f}%')

    else:
        st.write("Please enter some text to classify.")
