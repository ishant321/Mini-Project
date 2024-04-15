import streamlit as st
import pickle
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Twitter Disaster message classifier")
input_sms = st.text_area("Enter the message: ")

if st.button("Predict"):
    # 1. Preprocess
    # transformed_text = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([input_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Disaster Tweet")
    else:
        st.header("Not a Disaster Tweet")

