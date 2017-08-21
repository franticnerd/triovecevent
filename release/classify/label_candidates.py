import codecs
import json
import os.path
import sys

from zutils.config.param_handler import yaml_loader

'''
Label the candidates manually
'''
def label_candidates(description_file, label_file):
    candidates = load_candidate_descriptions(description_file)
    candidate_label_dict = load_candidate_labels(label_file)
    present_candidates(candidates, candidate_label_dict, label_file)

def load_candidate_descriptions(description_file):
    with codecs.open(description_file, 'r', 'utf-8') as fin:
        candidate_json = json.load(fin)
    candidates = {}
    for candidate in candidate_json:
        candidate_id = int(candidate['Id'])
        candidates[candidate_id] = candidate
    return candidates


def load_candidate_labels(label_file):
    if not os.path.exists(label_file):
        return {}
    candidate_label_dict = {}
    with open(label_file, 'r') as fin:
        for line in fin:
            items = line.strip().split('\t')
            candidate_id, label = int(items[0]), int(items[1])
            candidate_label_dict[candidate_id] = label
    return candidate_label_dict


def present_candidates(candidates, candidate_label_dict, label_file):
    unlabeled_candidate_ids = get_unlabeled_candidates(candidates, candidate_label_dict)
    while len(unlabeled_candidate_ids) > 0:
        candidate_id = unlabeled_candidate_ids.pop()
        present_one_candidate(candidates[candidate_id], candidate_label_dict)
        write_candidate_labels(candidate_label_dict, label_file)

def get_unlabeled_candidates(candidates, candidate_label_dict):
    unlabeled_candidates = set()
    for candidate_id in candidates:
        if candidate_id not in candidate_label_dict:
            unlabeled_candidates.add(candidate_id)
    return unlabeled_candidates

def present_one_candidate(candidate, candidate_label_dict):
    print 'Q: Do you think the following candidate is a local event?'
    print '\tRepresentative tweets:'
    for tweet in candidate['tweets']:
        print '\t\t\t*', tweet
    print 'Press \'y\' for YES, and \'n\' for NO.'
    candidate_id = int(candidate['Id'])
    while True:
        line = sys.stdin.readline()
        if line.strip() == 'y':
            candidate_label_dict[candidate_id] = 1
            return
        elif line.strip() == 'n':
            candidate_label_dict[candidate_id] = 0
            return
        else:
            print 'Please use \'y\' or \'n\' to indicate your choice.'


def write_candidate_labels(candidate_label_dict, label_file):
    with open(label_file, 'w') as fout:
        for k, v in candidate_label_dict.items():
            fout.write(str(k) + '\t' + str(v) + '\n')


if __name__ == '__main__':
    data_dir = '/Users/chao/Dropbox/data/class_event/sample/'
    if len(sys.argv) > 1:
        para_file = sys.argv[1]
        para = yaml_loader().load(para_file)
        data_dir = para['data_dir']
    description_file = data_dir + 'classify_candidate_descriptions.txt'
    label_file = data_dir + 'classify_candidate_labels.txt'
    label_candidates(description_file, label_file)
