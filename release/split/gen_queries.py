import operator
import random
import sys

from zutils.config.param_handler import yaml_loader
from zutils.datasets.twitter.tweet import Tweet


# build the mapping from file id to the time of the first tweet in that bin
def build_id_time_map(tweet_dir, setting_file):
    total_num_of_bin = load_total_num_of_bin(setting_file)
    id_time_map = {}
    for id in xrange(total_num_of_bin):
        tweet_file = tweet_dir + str(id) + '.tweet'
        timestamp = load_time_of_first_tweet(tweet_file)
        id_time_map[id] = timestamp
    return id_time_map


def load_total_num_of_bin(setting_file):
    with open(setting_file, 'r') as fin:
        line = fin.readline().strip()
        return int(line)


def load_time_of_first_tweet(tweet_file):
    with open(tweet_file, 'r') as fin:
        line = fin.readline().strip()
        tweet = Tweet()
        tweet.load_clean(line)
    return tweet.timestamp.timestamp


def gen_queries(id_time_map, num_query, query_length, ref_window_size):
    ret = []
    for i in xrange(num_query):
        query = gen_one_query(id_time_map, query_length, ref_window_size)
        ret.append(query)
    ret.sort( key = operator.itemgetter(0), reverse = False )
    return ret


def gen_one_query(id_time_map, query_length, ref_window_size):
    query_id = 0
    total_bin = len(id_time_map)
    while True:
        query_id = random.randint(ref_window_size, total_bin - query_length)
        tweet_time = id_time_map[query_id]
        if is_proper_time(tweet_time):
            break
    return [query_id + i for i in xrange(query_length)]


def is_proper_time(tweet_time):
    seconds = tweet_time % 86400
    return True if seconds >= 3600 * 11 and seconds <= 3600 * 22 else False


def write_queries(queries, query_file):
    with open(query_file, 'w') as fout:
        for q in queries:
            fout.write(' '.join([str(e) for e in q]) + '\n')


def run(input_tweet_dir, setting_file, query_file, num_query, query_length, ref_window_size):
    id_to_time = build_id_time_map(input_tweet_dir, setting_file)
    queries = gen_queries(id_to_time, num_query, query_length, ref_window_size)
    write_queries(queries, query_file)


if __name__ == '__main__':
    data_dir = '../../toy/'

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        num_query = 200
        query_length = 6
        ref_window_size = 14

    if 'toy' in data_dir:
        num_query = 2
        query_length = 6
        ref_window_size = 4

    tweet_dir = data_dir + 'tweets/'
    batch_info_file = data_dir + 'input/dataset_info.txt'
    query_file = data_dir + 'input/queries.txt'

    run(tweet_dir, batch_info_file, query_file, num_query, query_length, ref_window_size)
