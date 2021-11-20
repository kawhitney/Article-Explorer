import re
import ssl
import streamlit as st
import torch
from boilerpy3 import extractors
from transformers import pipeline

from question_answering import question_answering

# Run with: python -m streamlit run main.py


def run():
    # main process that creates sets up the streamlit app
    st.sidebar.title("Article Explorer")
    st.sidebar.subheader("Learn more about a text by asking it questions and reading its summary")
    article = None
    url = st.sidebar.text_input(label="url")
    if url:
        article = get_article(url)
        summary = summarize(article)
        st.write(summary)

    question = st.sidebar.text_input(label="Ask a question")
    if article and (question or (st.sidebar.button('Ask') and question)):
        st.write(answer(question, article))


def get_article(url):
    ssl._create_default_https_context = ssl._create_unverified_context
    # Fetches the text of an article from the given url
    text = extractors.ArticleExtractor(False).get_content_from_url(url)
    if 'wikipedia.org' in url:
        # Get rid of those pesky '[1]' footnotes from wikipedia articles
        text = re.sub(r"\[.*?\]+", '', text)
    return text


def summarize(text):
    # Extracts a summary from the text
    # currently grabs the first five sentences
    sents = text.split('.')
    if len(sents) >= 5:
        return sents[:5]
    else:
        return sents

# ================= Question Answering ========================
# https://towardsdatascience.com/question-and-answering-with-bert-6ef89a78dac

def answer(question, text):
    # get max number of tokens acceptable
    question_length = len(question)
    bert_text_length = 510 - question_length
    # get maximum length of text
    max_text_length = len(text)

    # class object
    qa = question_answering()

    # dictionary to hold answer and the prediction value
    result = {}

    for i in range(0, max_text_length, bert_text_length):
        # run BERT tokenization, store the start and end scores
        if i+bert_text_length < max_text_length:        # if length is more than text length
            prediciton = qa.predict_answer(question, text[i:i+bert_text_length])
        else:    # if length is less than text length
            prediciton = qa.predict_answer(question, text[i:max_text_length])
        # add result if higher prediciton value
        if not result or prediciton[1] > result["probability"]:
            result["answer"] = prediciton[0]
            result["probability"] = prediciton[1]

    return(result["answer"])




if __name__ == "__main__":
    run()