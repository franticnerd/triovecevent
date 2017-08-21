import sys

from tweet import Tweet

reload(sys)
sys.setdefaultencoding('utf-8')

class TweetDatabase:

    def __init__(self):
        self.tweets = []

    def load_clean_tweets_from_file(self, input_file):
        self.tweets = []
        # with codecs.open(input_file, 'r', 'utf-8') as fin:
        with open(input_file, 'r') as fin:
            for line in fin:
                tweet = Tweet()
                tweet.load_clean(line)
                self.tweets.append(tweet)

