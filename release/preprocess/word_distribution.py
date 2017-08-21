import operator
from math import log
from zutils.dto.st.grid import GridSpace

class WordEntropyProcessor:

    def __init__(self, tweet_database, grid_bin_list):
        self.td = tweet_database
        min_lat, max_lat = self.td.calc_lat_range()
        min_lng, max_lng = self.td.calc_lng_range()
        min_time, max_time = self.td.calc_time_range()
        self.grid_ranges = [(min_lat, max_lat), (min_lng, max_lng), (min_time, max_time)]
        self.grid_bins = grid_bin_list

    def calc(self, word_entropy_file):
        self.build_inverted_index()
        self.build_word_entropy_dict()
        self.rank_by_entropy()
        self.write_to_file(word_entropy_file)

    def select_top_words(self, fraction):
        num = int(len(self.word_entropy_list) * fraction)
        return set([t[0] for t in self.word_entropy_list[:num]])

    def build_inverted_index(self):
        print 'Begin building inverted index...'
        self.inverted_index = {}
        for i, tweet in enumerate(self.td.tweets):
            for word in tweet.message.words:
                l = self.inverted_index.get(word, [])
                l.append(i)
                self.inverted_index[word] = l


    def build_word_entropy_dict(self):
        print 'Begin computing word entropy...'
        self.word_entropy_dict = {}
        cnt = 0
        for word, tweet_indices in self.inverted_index.items():
            # print word
            # print tweet_indices
            entropy = self.calc_word_entropy(tweet_indices)
            self.word_entropy_dict[word] = entropy
            cnt += 1
            if cnt % 100 == 0:
                print cnt


    def rank_by_entropy(self):
        self.word_entropy_list = []
        for w, e in self.word_entropy_dict.items():
            self.word_entropy_list.append((w, e))
        self.word_entropy_list.sort(key = operator.itemgetter(1), reverse=True)


    def calc_word_entropy(self, tweet_indices, freq_thresh=40000):
        dist = self.calc_word_distributions(tweet_indices)
        frequency = dist.get_l1_norm()
        # print frequency
        # for k, v in dist.data.items():
        #     print k, v
        # KL divergence
        if frequency < freq_thresh:
            localness = log(frequency) - dist.get_entropy()
        else:
            localness = log(freq_thresh) - dist.get_entropy()
        return localness


    def calc_word_distributions(self, tweet_indices):
        grid = GridSpace(self.grid_ranges, self.grid_bins)
        # print self.grid_ranges
        dist = Distribution()
        for tweet_index in tweet_indices:
            t = self.td.tweets[tweet_index]
            # print tweet_index, t
            dim = grid.get_cell_id([t.location.lat, t.location.lng, t.timestamp.timestamp])
            dist.add_value(dim, tweet_index)
        return dist


    def write_to_file(self, out_file):
        with open(out_file, 'w') as fout:
            for word, localness in self.word_entropy_list:
                fout.write(word + ',' + str(localness) + '\n')



class Distribution:

    def __init__(self, length=None):
        self.L = length
        self.data = {}

    # add the value for the given dimension
    def add_value(self, dim, value):
        old_value = self.data.get(dim, set())
        old_value.add(value)
        self.data[dim] = old_value

    def get_l1_norm(self):
        ret = 0
        for key in self.data:
            ret += len(self.data[key])
        return ret

    def normalize(self):
        l1_norm = self.get_l1_norm()
        for key in self.data:
            self.data[key] = float(len(self.data[key])) / float(l1_norm)

    # get the entroy for the probability distribution encoded by current vector
    def get_entropy(self):
        ret = 0
        self.normalize()
        for value in self.data.values():
            if value <= 1e-20:
                continue
            ret -= value * log(value)
        return ret

    # convert to dict object
    def to_dict(self):
        ret = copy.deepcopy(self.data)
        ret['L'] = self.L
        return ret

    def load_from_dict(self, d):
        self.L = d['L']
        self.data = copy.deepcopy(d)
        del self.data['L']


if __name__ == '__main__':
    pass
