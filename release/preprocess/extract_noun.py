import sys
import os

from zutils.datasets.twitter.filters import ContainWordFilter
from zutils.datasets.twitter.filters import EmptyMessageFilter
from zutils.datasets.twitter.pos_database import PosDatabase
from zutils.datasets.twitter.tweet_database import TweetDatabase

from params import *
from zutils.dto.text.word_distribution import WordEntropyProcessor


def load_tweet_database(tweet_file):
    td = TweetDatabase()
    td.load_clean_tweets_from_file(tweet_file)
    return td


def load_postags(pos_tag_file, entity_file):
    wd = PosDatabase()
    wd.load_postags(pos_tag_file)
    wd.load_entities(entity_file)
    return wd


def filter_by_pos_types(td, wd, comb, vocab_file, freq_thre, infreq_thre):
    wd.replace_keywords(td, comb)
    td.trim_words_by_frequency(vocab_file, freq_thre, infreq_thre)
    emf = EmptyMessageFilter()
    td.apply_one_filter(emf)


def filter_activity_tweets(td, word_entropy_file, activity_word_fraction):
    wep = WordEntropyProcessor(td)
    wep.calc(word_entropy_file)
    activity_words = wep.select_top_words(activity_word_fraction)
    cwf = ContainWordFilter(activity_words)
    td.apply_one_filter(cwf)


def run(pd):
    raw_tweet_file = pd['raw_data_dir'] + 'tweets.txt'
    td = load_tweet_database(raw_tweet_file)

    postag_file = pd['raw_data_dir'] + 'postags.txt'
    entity_file = pd['raw_data_dir'] + 'entities.txt'
    wd = load_postags(postag_file, entity_file)

    comb = pd['pos_types']
    freq_thre = pd['freq_thre']
    infreq_thre = pd['infreq_thre']
    vocab_file = pd['data_dir'] + 'input/vocab.txt'
    filter_by_pos_types(td, wd, comb, vocab_file, freq_thre, infreq_thre)

    activity_word_fraction = 0.3
    word_entropy_file = pd['data_dir'] + 'input/word_concentration.txt'
    directory = os.path.dirname(word_entropy_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filter_activity_tweets(td, word_entropy_file, activity_word_fraction)

    clean_tweet_file = pd['data_dir'] + 'input/tweets.txt'
    directory = os.path.dirname(clean_tweet_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    td.write_clean_tweets_to_file(clean_tweet_file)


if __name__ == '__main__':
    dataset = 'toy' if len(sys.argv) <= 1 else sys.argv[1]
    if dataset == 'toy':
        pd = load_toy_params()
    elif dataset == 'la':
        pd = load_la_params()
    elif dataset == 'ny':
        pd = load_ny_params()
    run(pd)
