import codecs
import json
import operator
import sys
import time
import urllib2
from itertools import groupby

import numpy as np
import numpy.linalg as npla
from scipy.spatial.distance import cosine


# load all the candidates, without filtering or ranking
def load_candidates(candidate_file):
    candidates = []
    with open(candidate_file) as fin:
        for line in fin:
            candidate = json.loads(line)
            candidates.append(candidate)
    print 'Number of raw candidates:', len(candidates)
    return candidates


'''
Module I. generate the features for each candidate;
'''
def gen_candidate_features(candidates):
    features = []
    for candidate_id, candidate in enumerate(candidates):
        tweets = candidate['members']
        # print len(tweets)
        feature = []
        feature.append(get_spatial_unusualness(tweets))
        feature.append(get_temporal_unusualness(tweets))
        feature.append(get_spatiotemporal_unusualness(tweets))
        feature.extend(get_spatial_concentration(tweets))
        feature.append(get_temporal_concentration(tweets))
        feature.append(get_textual_concentration(tweets))
        # feature.append(get_uniqueness(candidate, candidates))
        feature.append(get_spatial_burstiness(tweets))
        features.append(feature)
    return zip(candidates, features)


def write_candidate_features(candidate_features, candidate_feature_file):
    with open(candidate_feature_file, 'w') as fout:
        fout.write('\t'.join(['spatial_unusualness',
                              'temporal_unusualness',
                              'st_unusualness',
                              'lat_std',
                              'lng_std',
                              'time_std',
                              'semantic_std',
                              # 'uniqueness',
                              'spatial_burstiness'
                              ]) + '\n')
        for (candidate, feature) in candidate_features:
            fout.write('\t'.join([str(e) for e in feature]) + '\n')


def filter_and_rank(candidate_features):
    ret = []
    for key, group in groupby(candidate_features, lambda x: x[0]['query_id']):
        # print 'query_id', key
        # filter the results for one query
        ret.extend(filter_and_rank_for_one_query(group))
    ret = sorted(ret, key=lambda c : c[1][2], reverse=False)
    return ret


# limit: the maximum number of candidates in each query window, default: 10
def filter_and_rank_for_one_query(candidate_features, limit = 10):
    # filter the candidates that are too spread
    candidate_features = [cf for cf in candidate_features if (cf[1][3] < 0.02 and cf[1][4] < 0.02)]
    # sort by the spatiotemporal unusualness, in descending order
    candidate_features = sorted(candidate_features, key=lambda c : c[1][2], reverse=False)
    # limit the total number of candidates
    return candidate_features[:limit]


def get_spatial_unusualness(tweets):
    cosines = [calc_cosine(t['spatial_embedding'], t['textual_embedding']) for t in tweets]
    return np.mean(cosines)


def get_temporal_unusualness(tweets):
    cosines = [calc_cosine(t['temporal_embedding'], t['textual_embedding']) for t in tweets]
    return np.mean(cosines)


def get_spatiotemporal_unusualness(tweets):
    cosines = []
    for t in tweets:
        avg_st_embedding = (np.array(t['spatial_embedding']) + np.array(t['temporal_embedding'])) / 2
        cosines.append(calc_cosine(avg_st_embedding, t['textual_embedding']))
        # s_cosine = calc_cosine(t['spatial_embedding'], t['textual_embedding'])
        # t_cosine = calc_cosine(t['temporal_embedding'], t['textual_embedding'])
        # cosines.append((s_cosine + t_cosine) / 2.0)
    return np.mean(cosines)


def calc_cosine(e1, e2):
    return 1.0 - cosine(e1, e2)


# def get_spatial_burstiness(tweets):
#     user_set = set()
#     for t in tweets:
#         user_set.add(t['user_id'])
#     return len(user_set)

def get_spatial_burstiness(tweets):
    return len(tweets)

def get_temporal_burstiness(tweets):
    return len(tweets)


def get_spatiotemporal_burstiness(tweets):
    return len(tweets)


def get_spatial_concentration(tweets):
    lats = np.array([t['lat'] for t in tweets])
    lngs = np.array([t['lng'] for t in tweets])
    return np.std(lats), np.std(lngs)


def get_temporal_concentration(tweets):
    timestamps = np.array([t['timestamp'] for t in tweets])
    return np.std(timestamps)


def get_textual_concentration(tweets):
    mean_vector = get_mean_direction(tweets)
    # for t in tweets:
        # print mean_vector
        # print t['textual_embedding']
    cosines = [calc_cosine(mean_vector, t['textual_embedding']) for t in tweets]
    return np.mean(cosines)

# def get_textual_concentration(tweets):
#     cosines = [calc_cosine(t1['textual_embedding'], t2['textual_embedding']) for t1 in tweets for t2 in tweets]
#     return np.mean(cosines)


def get_uniqueness(candidate, candidates):
    query_id = candidate['query_id']
    same_query_candidates = [c for c in candidates if c['query_id'] == query_id]
    cosines = []
    for c in same_query_candidates:
        v1 = get_mean_direction(candidate['members'])
        v2 = get_mean_direction(c['members'])
        if v1 is not None and v2 is not None:
            cosines.append(calc_cosine(v1, v2))
    # remove itself from the average computation
    return (sum(cosines) - 1.0) / (len(cosines) - 1)



def get_mean_direction(tweets):
    normed_vectors = []
    for t in tweets:
        v = np.array(t['textual_embedding'])
        if npla.norm(v, 2) == 0:
            print v
            continue
        normed_vectors.append(v / npla.norm(v, 2))
    # print sum(normed_vectors) / float(len(normed_vectors))
    if len(normed_vectors) == 0:
        return None
    return sum(normed_vectors) / float(len(normed_vectors))



# convert a dict to a vector, representing the probability distribution
def to_vector(dim, values):
    ret = np.zeros(dim)
    for k, v in values.items():
        ret[int(k)] = float(v)
    return ret





'''
Module II. generate the description of each candidate.
'''
def gen_candidate_locations(candidates, candidate_location_dir, filter_file = None):
    if filter_file is not None:
        filtered_candidates = load_true_events_ids(filter_file)
    for i, candidate in enumerate(candidates):
        if filter_file is not None and i not in filtered_candidates:
            continue
        print 'plotting location for candidate ', i
        tweets = candidate['members']
        locations = [(t['lat'], t['lng']) for t in tweets]
        request ='https://maps.googleapis.com/maps/api/staticmap?zoom=10&size=600x600&maptype=roadmap&'
        for lat, lng in locations:
            request += 'markers=color:red%7C' + '%f,%f&' % (lat, lng)
        proxy = urllib2.ProxyHandler({'https': '104.197.200.10:80'})
        opener = urllib2.build_opener(proxy)
        response = opener.open(request).read()
        output_path = candidate_location_dir + str(i) + '.png'
        with open(output_path, 'wb') as f:
            f.write(response)
            f.close()
        time.sleep(3)


def load_true_events_ids(filter_file):
    true_event_ids = set()
    with open(filter_file, 'r') as fin:
        for line in fin:
            items = line.strip().split()
            if int(items[1]) == 1:
                true_event_ids.add(int(items[0]))
    return true_event_ids



def gen_candidate_descriptions(candidates, candidate_description_file):
    descriptions = []
    for i, candidate in enumerate(candidates):
        # print 'number of members', len(candidate['members'])
        one_candidate = {'Id': i, 'Size:': len(candidate['members'])}
        one_candidate.update(get_top_tweets(candidate))
        descriptions.append(one_candidate)
    with codecs.open(candidate_description_file, 'w', 'utf-8') as fout:
        fout.write(json.dumps(descriptions, indent = 2))


# get the top N tweets for the candidate, return a dict
def get_top_tweets(candidate, num=10):
    tweets = candidate['members']
    mean_vector = get_mean_direction(tweets)
    # sort the tweets by authority
    sort_list = []
    for t in tweets:
        score = calc_cosine(mean_vector, t['textual_embedding'])
        sort_list.append((t['message'], score))
    sort_list.sort( key = operator.itemgetter(1), reverse = True )
    # retrieve the tweet text from the database
    tweet_text = []
    for message, score in sort_list[:num]:
        tweet_text.append(message)
    return {'tweets': tweet_text}



def run(candidate_file, feature_file, description_file, location_dir, filter_file=None):
    raw_candidates = load_candidates(candidate_file)
    start = time.time()
    candidate_features = gen_candidate_features(raw_candidates)
    candidate_features = filter_and_rank(candidate_features)
    end = time.time()
    print 'Time for feature extraction: ', end - start
    print 'Number of filtered candidates:', len(candidate_features)
    write_candidate_features(candidate_features, feature_file)
    # candidates = [e[0] for e in candidate_features]
    # gen_candidate_descriptions(candidates, description_file)
    # gen_candidate_locations(candidates, location_dir, filter_file)


if __name__ == '__main__':
    data_dir = '/Users/chao/data/projects/triovecevent/toy/'
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    classify_dir = data_dir + 'classify/'
    candidate_file = classify_dir + 'candidates.txt'
    feature_file = classify_dir + 'classify_candidate_features.txt'
    description_file = classify_dir + 'classify_candidate_descriptions.txt'
    location_dir = classify_dir + '/locations/'
    filter_file = classify_dir + 'classify_candidate_labels.txt'
    run(candidate_file, feature_file, description_file, location_dir, filter_file)

