# https://towardsdatascience.com/build-an-open-domain-question-answering-system-with-bert-in-3-lines-of-code-da0131bc516b
# open-domain QA
import ktrain
from ktrain import text
import shutil

# load dataset into an array - NEED TO SWITCH OUT TO SQuAD
from sklearn.datasets import fetch_20newsgroups
remove = ('headers', 'footers', 'quotes')
newsgroups_train = fetch_20newsgroups(subset='train', remove=remove)
newsgroups_test = fetch_20newsgroups(subset='test', remove=remove)
docs = newsgroups_train.data + newsgroups_test.data

# create a search index
INDEXDIR = '/tmp/myindex'
text.SimpleQA.initialize_index(INDEXDIR)
text.SimpleQA.index_from_list(docs, INDEXDIR, commit_every=len(docs))
# if document set too large, use: Simple.QA.index_from_folder

# create a QA instance
qa = text.SimpleQA(INDEXDIR)

# ask the question
answers = qa.ask('When did the Cassini probe launch?')
qa.display_answers(answers[:5])
plt.show()

# print top answer
print(docs[59])

# remove temp directory
shutil.rmtree(INDEXDIR)