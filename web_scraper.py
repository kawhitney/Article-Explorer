# web scraper utilizing beautiful soup
import urllib.request
from bs4 import BeautifulSoup
import nltk

# stored web page value
webpage = 'https://www.ibm.com/cloud/learn/natural-language-processing'

# get data from web pages
response = urllib.request.urlopen(webpage)
html = response.read()

# use beautiful soup and html5lib to clean out the html
soup = BeautifulSoup(html,"html5lib")
text = soup.find('article').get_text(strip=True)

print(f"{text}")