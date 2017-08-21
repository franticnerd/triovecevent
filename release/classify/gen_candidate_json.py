import ast
import codecs
import json
import os.path
import sys

import numpy.linalg as npla

from twitter.tweet_database import TweetDatabase


# from zutils.datasets.twitter.tweet_database import TweetDatabase

# each query: (query_id, bin_id_list)
def load_queries(query_file):
    queries = []
    with open(query_file, 'r') as fin:
        lines = fin.readlines()
    for (query_id, line) in enumerate(lines):
        items = line.strip().split()
        bin_id_list = [int(e) for e in items]
        queries.append((query_id, bin_id_list))
    return queries


# def load_clustering_results(queries, tweet_dir, clustering_dir):
#     results = []
#     for query in queries:
#         try:
#             r = load_results_for_one_query(query, tweet_dir, clustering_dir)
#             results.append(r)
#         except:
#             print 'Exception in loading query results for query ', query
#             return results
#     return results


def load_clustering_results(queries, tweet_dir, clustering_dir):
    results = []
    for query in queries:
        r = load_results_for_one_query(query, tweet_dir, clustering_dir)
        results.append(r)
    return results


def load_results_for_one_query(query, tweet_dir, clustering_dir):
    tweets = load_tweets(query, tweet_dir)
    clustering = load_clustering_membership(query, clustering_dir)
    return (query, tweets, clustering)


def load_tweets(query, tweet_dir):
    tweets = []
    for bin_id in query[1]:
        tweet_file = tweet_dir + str(bin_id) + '.tweet'
        tdb = TweetDatabase()
        tdb.load_clean_tweets_from_file(tweet_file)
        tweets.extend(tdb.tweets)
    return tweets
    # return [t.tid for t in tweets]


def load_clustering_membership(query, clustering_dir):
    membership = []
    query_id = query[0]
    for bin_id in query[1]:
        result_file = clustering_dir + str(query_id) + '-' + str(bin_id) + '.csv'
        with open(result_file, 'r') as fin:
            for line in fin:
                items = line.strip().split(',')
                membership.append(int(items[-1]))
    return membership


# read the embeddings that correspond to the tweets in the clustering results
def load_embeddings(clustering_results, embedding_file):
    tweet_ids = extract_tweet_ids(clustering_results)
    return read_embedding_with_filtering(embedding_file, tweet_ids)


def extract_tweet_ids(clustering_results):
    tweet_ids = set()
    for c in clustering_results:
        tweets = c[1]
        for tweet in tweets:
            tweet_ids.add(tweet.tid)
    return tweet_ids


def read_embedding_with_filtering(embedding_file, tweet_ids):
    embeddings = {}
    with open(embedding_file, 'r') as fin:
        for line in fin:
            fields = line.strip().split('\x01')
            tweet_id = long(fields[0])
            if tweet_id in tweet_ids:
                spatial_feature, temporal_feature, textual_feature = [ast.literal_eval(field) for field in fields[1:]]
                embeddings[tweet_id] = [spatial_feature, temporal_feature, textual_feature]
    return embeddings


# write the clustering results and features
def make_json_for_candidates(clustering_results, embeddings):
    all_candidates = []
    for one_query_result in clustering_results:
        query, tweets, clustering = one_query_result
        j = make_json_for_one_query(query, clustering, tweets, embeddings)
        all_candidates.extend(j)
    return all_candidates


def make_json_for_one_query(query, clustering, tweets, embeddings):
    cluster_tweet_map = build_cluster_tweet_map(tweets, clustering)
    candidates = []
    for cluster_id, members in cluster_tweet_map.items():
        num_user = get_num_of_users(members)
        # print cluster_id, num_user
        # if num_user >= 5 and num_user < 50:
        if num_user >= 5 and num_user < 50:
            member_tweet_list = []
            for tweet in members:
                tweet_json = make_json_for_one_tweet(tweet, embeddings)
                if npla.norm(tweet_json['textual_embedding'], 2) != 0:
                    member_tweet_list.append(tweet_json)
            d = {'query_id': query[0], 'query_bin_list': query[1], 'members': member_tweet_list}
            if len(member_tweet_list) >= 5:
                candidates.append(d)
    return candidates


def get_num_of_users(members):
    user_set = set()
    for tweet in members:
        user_set.add(tweet.uid)
    return len(user_set)


def build_cluster_tweet_map(tweets, clusters):
    ret = {}
    for (tweet, cluster) in zip(tweets, clusters):
        if cluster not in ret:
            members = [tweet]
            ret[cluster] = members
        else:
            ret[cluster].append(tweet)
    return ret


def make_json_for_one_tweet(tweet, embeddings):
    tweet_id = tweet.tid
    spatial_embedding, temporal_embedding, textual_embedding = embeddings[tweet_id]
    ret = {'tweet_id': tweet.tid, 'user_id': tweet.uid, 'lat': tweet.location.lat, 'lng': tweet.location.lng, \
           'created_time': tweet.timestamp.time_string, 'timestamp': tweet.timestamp.timestamp, \
           'message': tweet.message.raw_message.encode('utf-8'), \
           'spatial_embedding': spatial_embedding, 'temporal_embedding': temporal_embedding, \
           'textual_embedding': textual_embedding}
    return ret



def write_to_file(candidates, output_file):
    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with codecs.open(output_file, 'w', 'utf-8') as fout:
        for c in candidates:
            fout.write(json.dumps(c) + '\n')


def run(query_file, tweet_dir, clustering_dir, embedding_file, candidates_file):
    queries = load_queries(query_file)
    clustering_results = load_clustering_results(queries, tweet_dir, clustering_dir)
    embeddings = load_embeddings(clustering_results, embedding_file)
    candidates = make_json_for_candidates(clustering_results, embeddings)
    write_to_file(candidates, candidates_file)


if __name__ == '__main__':
    # sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
    data_dir = '/Users/chao/data/projects/triovecevent/toy/'
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    query_file = data_dir + 'input/queries.txt'
    tweet_dir = data_dir + 'tweets/'
    cluster_dir = data_dir + 'cluster/'
    embedding_file = data_dir + 'embeddings/embeddings.txt'
    candidate_file = data_dir + 'classify/candidates.txt'
    run(query_file, tweet_dir, cluster_dir, embedding_file, candidate_file)
