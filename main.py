import re
import ssl
import torch
import streamlit as st
from boilerpy3 import extractors
from summarizer import Summarizer


# Run with python -m streamlit main.py
# Could also be run with streamlit run main.py


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

        # text = text.find('\n\n',text)

    return text


def summarize(text):
    # Extracts a summary from the text
    model = Summarizer()
    result = model(text, num_sentences=5)
    return result


def answer(question, text):
    # Answers a question about the text
    # currently ignores the question and returns the last sentence
    return text.split('.')[-1]


if __name__ == "__main__":
    run()
