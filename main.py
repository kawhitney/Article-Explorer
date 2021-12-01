import re
from io import StringIO
import streamlit as st
from boilerpy3 import extractors

# Run with python -m streamlit main.py


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
    # Fetches the text of an article from the given url
    text = extractors.ArticleExtractor().get_content_from_url(url)
    print(text)
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


def answer(question, text):
    # Answers a question about the text
    # currenty ignores the question and returns the last sentence
    return text.split('.')[-1]


if __name__ == "__main__":
    run()
