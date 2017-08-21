from collections import Counter
from zutils.algo.utils import ensure_directory_exist
from zutils.algo.utils import format_list_to_string

class WordDict:

    def __init__(self):
        self.word_counter = Counter()  # the counter of the words
        self.word_to_id = {}  # key: word, value: word_id
        self.id_to_word = {}  # key: id, value: word

    def size(self):
        return len(self.id_to_word)

    # update the support of different words, input is a set/list of words
    def update_count(self, words):
        self.word_counter.update(words)

    # encode the words by support, starting word_id is 0
    def encode_words(self, min_freq=1):
        ranked_words = self.word_counter.most_common()
        word_id = 0
        for (word, count) in ranked_words:
            # skip all the words appear less than min_freq times
            if count < min_freq:
                break
            self.id_to_word[word_id] = word
            self.word_to_id[word] = word_id
            word_id += 1


    # get the word from id
    def get_word(self, id):
        return self.id_to_word[id]

    # get the id of a word
    def get_word_id(self, word):
        return self.word_to_id[word]

    # convert words to ids, ignore the words that are not in vocabulory
    def convert_to_ids(self, words):
        ret = []
        for w in words:
            if w in self.word_to_id:
                ret.append(self.word_to_id[w])
        return ret

    # get the count of a word
    def get_word_cnt(self, word):
        return self.word_counter[word]

    # given a threshold, return all the words that are infrequent
    def get_infrequent_words(self, threshold):
        ret = set()
        for w, c in self.word_counter.items():
            if c <= threshold:
                ret.add(w)
        return ret

    # given a threshold, return all the words that are frequent
    def get_frequent_words(self, threshold):
        ret = set()
        for w, c in self.word_counter.items():
            if c >= threshold:
                ret.add(w)
        return ret

    # write the word info into a file
    def write_to_file(self, output_file):
        ensure_directory_exist(output_file)
        with open(output_file, 'w') as fout:
            for word_id in xrange(len(self.id_to_word)):
                word = self.id_to_word[word_id]
                count = self.word_counter[word]
                fout.write(format_list_to_string([word_id, word, count]) + '\n')


    # load a word dict from the file
    def load_from_file(self, input_file, sep = '\t'):
        self.word_counter = Counter()
        self.word_to_id = {}
        self.id_to_word = {}
        with open(input_file, 'r') as fin:
            for line in fin:
                items = line.strip().split(sep)
                word_id = int(items[0])
                word = str(items[1])
                self.word_to_id[word] = word_id
                self.id_to_word[word_id] = word
                if len(items) > 2:
                    word_cnt = int(items[2])
                    self.word_counter[word] = word_cnt


                    # # return the words by support, and set the field self.words
                    # def rank_words(self):
                    #     ranked = self.word_counter.most_common()
                    #     self.words = [e[0] for e in ranked]

                    # # construct all fields from scratch, input is a list of word list
                    # def generate(self, word_lists):
                    #     for l in word_lists:
                    #         self.update_count(l)
                    #     self.encode_words()

if __name__ == '__main__':
    wd = WordDict()
    wd.update_count(['hello', 'world'])
    wd.update_count(['world'])
    wd.encode_words()
    print wd.word_counter
    print wd.word_to_id
    print wd.id_to_word
    wd.write_to_file('/Users/chao/Downloads/test.txt')
