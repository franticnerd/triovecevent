from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import CMUTweetTagger
import re


class TextParser:

    def __init__(self, min_length=2, stopword_file=None):
        self.min_length = min_length
        self.stopwords = set(stopwords.words('english'))
        if stopword_file is not None:
            self.load_stopwords(stopword_file)

    '''
    Allow a user to specify stopwords through a file; each line is a stop word.
    '''
    def load_stopwords(self, stopword_file):
        with open(stopword_file, 'r') as fin:
            for line in fin:
                stopword = line.strip()
                self.stopwords.add(stopword)

    '''
    Parse a string into a list of words. Perform stemming and stopword removal.
    '''
    def parse_words(self, input_string, stem=True):
        word_pattern = re.compile(r'[0-9a-zA-Z]+')
        words = []
        tokens = re.findall(word_pattern, input_string)
        wnl = WordNetLemmatizer()
        for token in tokens:
            if len(token) < self.min_length:
                continue
            word = wnl.lemmatize(token.lower()) if stem else token.lower()
            if self._is_valid(word):
                words.append(word)
        return words


    def _is_valid(self, word):
        #  Check whether the word is too short
        if(len(word) < self.min_length):
            return False
        #  Check whether the word is a stop word
        if(word in self.stopwords):
            return False
        return True


    '''
    Parse a string into a list of tokens using the ark tweet nlp tool
    '''
    def parse_words_by_ark_nlp(self, tweet, preserve_types, ark_run_cmd):
        tokens = self.tokenize_by_ark_nlp(tweet, ark_run_cmd)
        filtered_tokens = self.filter_by_type(tokens, preserve_types)
        words = self.parse_tokenized_words(filtered_tokens)
        return words

    # Use ark_tweet_nlp to parse a raw tweet message into a list of tokens.
    def tokenize_by_ark_nlp(self, tweet, ark_run_cmd):
        token_lists = CMUTweetTagger.runtagger_parse([tweet.strip()], run_tagger_cmd=ark_run_cmd)
        return token_lists[0]

    def filter_by_type(self, tokens, preserve_types):
        ret = []
        for w in tokens:
            if w[1] in preserve_types:
                ret.append(w)
        return ret

    def parse_tokenized_words(self, tokens):
        s = ' '.join([w[0] for w in tokens])
        return self.parse_words(s)


    # Use ark_tweet_nlp to parse a set of raw tweet messages
    def parse_words_by_ark_nlp_batch(self, tweets, preserve_types, ark_run_cmd):
        token_lists = CMUTweetTagger.runtagger_parse(tweets, run_tagger_cmd=ark_run_cmd)
        ret = []
        for tokens in token_lists:
            filtered_tokens = self.filter_by_type(tokens, preserve_types)
            words = self.parse_tokenized_words(filtered_tokens)
            ret.append(words)
        return ret

    # Use ark_tweet_nlp to parse a set of raw tweet messages and return all tags
    def get_pos_tag_lists(self, tweets, ark_run_cmd):
        token_lists = CMUTweetTagger.runtagger_parse(tweets, run_tagger_cmd=ark_run_cmd)
        return token_lists

if __name__ == '__main__':
    s1 = 'hello, 13423, This is @ZC went octopi a test for 12you!. Try it http://'
    s2 = '@ZC I love New York!!!!'
    wp = TextProcessor(min_length = 2)
    ark_run_cmd='java -XX:ParallelGCThreads=2 -Xmx2G -jar /Users/chao/Dropbox/code/lib/ark-tweet-nlp-0.3.2.jar'
    print wp.parse_words(s1)
    print wp.parse_words_by_ark_nlp(s1, set(['S', 'N', '^']), ark_run_cmd)

