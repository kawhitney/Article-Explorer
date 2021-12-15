import re
from io import StringIO
import ssl
import streamlit as st
from boilerpy3 import extractors
from summarizer import Summarizer
from question_answering import QuestionAnswering

# Run with: python -m streamlit run main.py
# Could also be run with streamlit run main.py


def run():
    # main process that creates sets up the streamlit app
    st.sidebar.title("Article Explorer")
    st.sidebar.subheader("Learn more about a text by asking it questions and reading its summary")
    article = None
    summary = None
    url = st.sidebar.text_input(label="Enter the Url to an article")
    uploaded_file = st.sidebar.file_uploader(label="Or upload a text file", type='txt')

    if uploaded_file is not None:
        article = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    if url:
        article = get_article(url)

    if article:
        article = clean_text(article)
        if not summary:
            summary = summarize(article)
            st.write(summary)
        question = st.sidebar.text_input(label="Ask a question")
        btn = st.sidebar.button('Ask')
        if question or (btn and question):
            print(article.split('.'))
            st.write("Answer: " + answer(question, article))


def clean_text(text):
    # remove extra whitespace
    return ".".join([sent.strip() for sent in text.split('.')])


def get_article(url):
    ssl._create_default_https_context = ssl._create_unverified_context
    # Fetches the text of an article from the given url
    text = extractors.ArticleExtractor(False).get_content_from_url(url)
    if 'wikipedia.org' in url:
        # Get rid of those pesky '[1]' footnotes from wikipedia articles
        text = re.sub(r"\[.*?\]+", '', text)
    return text


# ================= Summarization ========================
# https://pypi.org/project/bert-extractive-summarizer/

def summarize(text):
    # Extracts a summary from the text
    model = Summarizer()
    result = model(text, num_sentences=5)
    return result


# ================= Question Answering ========================
# https://towardsdatascience.com/question-and-answering-with-bert-6ef89a78dac


def answer(question, text):
    # get max number of tokens acceptable
    question_length = len(question)
    bert_text_length = 510 - question_length
    # get maximum length of text
    max_text_length = len(text)

    # class object
    qa = QuestionAnswering()

    # dictionary to hold answer and the prediction value
    result = {}

    for i in range(0, max_text_length, bert_text_length-100):
        # run BERT tokenization, store the start and end scores
        if i+bert_text_length < max_text_length:        # if length is more than text length
            prediction = qa.predict_answer(question, text[i:i+bert_text_length])
        else:    # if length is less than max text length
            prediction = qa.predict_answer(question, text[i:max_text_length])
        # add result if higher prediciton value
        if not result or prediction[1] >= result["probability"]:
            result["answer"] = prediction[0]
            result["probability"] = prediction[1]
    if result["probability"] < 2:
        result["answer"] = "No answer was found for this question."
    
    return result["answer"]


if __name__ == "__main__":
    run()
