from gensim.test.utils import common_corpus, common_dictionary
from gensim.models import HdpModel
hdp = HdpModel(common_corpus, common_dictionary)
unseen_doc = [(1, 3.), (2, 4)]
doc_hdp = hdp[unseen_doc]
print(doc_hdp)

topics = hdp.get_topics()
num_topics, vocab_size = topics.shape
print("Found {} topics.".format(num_topics))
print("Printing top 20 topics")
topic_info = hdp.print_topics(num_topics=20, num_words=20)
for _, topic in enumerate(topic_info):
    num, words = topic
    print("Topic {}: {}".format(num, words))
