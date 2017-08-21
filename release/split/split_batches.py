import ast
import json
import numpy as np 
import argparse
import os
import sys
import csv
import datetime
import calendar
from dateutil.parser import parse

parser = argparse.ArgumentParser()
# parser.add_argument("--tweets", help = "input file", default = "/shared/data/czhang82/clean/ny_tweets/tweets.txt")
# parser.add_argument("--embeddings", help = "input file", default = "/shared/data/kzhang53/embedding/ny.txt")
# parser.add_argument("--output_embedding_folder", default='/shared/data/ll2/tweets/online/embed/')
# parser.add_argument("--output_tweets_folder", default='/shared/data/ll2/tweets/online/tweets/')
# parser.add_argument("--output_setting", default='/shared/data/ll2/tweets/online/setting.csv')
parser.add_argument("--data_dir", help = 'data directory', default = '/Users/chao/data/projects/triovecevent/toy/')
parser.add_argument("--delimiter", help = 'del', default = '\x01')
parser.add_argument("--inter_val", type = int, default = 30)
args = parser.parse_args()

args.tweets = args.data_dir + 'input/tweets.txt'
args.embeddings = args.data_dir + 'embeddings/embeddings.txt'
args.output_tweets_folder = args.data_dir + 'tweets/'
args.output_embedding_folder = args.data_dir + 'embeddings/'
args.output_setting = args.data_dir + 'input/dataset_info.txt'


if __name__ == "__main__":

	month_dict = {v: k for k,v in enumerate(calendar.month_abbr)}

	def gettime(ins):
		# nins = ins.split(args.delimiter)
		timestring = ins[2]
		return parse(timestring)
		# timestring = timestring.split(' ')
		# hourstring = timestring[3].split(':')
		# return datetime.datetime(int(timestring[5]), month_dict[timestring[1]], int(timestring[2]), int(hourstring[0]), int(hourstring[1]), int(hourstring[2]))#, float(nins[2]), float(nins[3])

	with open(args.tweets, 'r') as f:
		tweets = f.readlines()

	inses = map(lambda t: t.split(args.delimiter)[2:5], tweets)
	sum0 = 0.0
	sum1 = 0.0
	for idx in range(0, len(inses)):
		inses[idx][0] = float(inses[idx][0])
		sum0 += inses[idx][0]
		inses[idx][1] = float(inses[idx][1])
		sum1 += inses[idx][1]
	sum0 /= len(inses)
	sum1 /= len(inses)
	ssum0 = 0.0
	ssum1 = 0.0
	print "average"
	for ins in inses:
		ssum0 += (ins[0] - sum0) ** 2;
		ssum1 += (ins[1] - sum1) ** 2;
	ssum0 /= (len(inses) - 1)
	ssum1 /= (len(inses) - 1)
	ssum1 = np.sqrt(ssum1)
	ssum0 = np.sqrt(ssum0)
	print "std"
	for idx in range(0, len(inses)):
		inses[idx][0] = (inses[idx][0] - sum0) / ssum0
		inses[idx][1] = (inses[idx][1] - sum1) / ssum1
		
	print "loading done"
		
	inverval_sec = args.inter_val*60

        if not os.path.exists(args.output_embedding_folder):
            os.makedirs(args.output_embedding_folder)
        if not os.path.exists(args.output_tweets_folder):
            os.makedirs(args.output_tweets_folder)

	# length = min(len(inses), len(embeddings))
	cur_time = gettime(inses[0])
	find = 0
	embed_out = open(args.output_embedding_folder+str(find)+'.embed', 'w')
	tweet_out = open(args.output_tweets_folder+str(find)+'.tweet', 'w')
	length_map = {}
	embed_len = 0
	cur_length = 0

	def output_norm_vec(tmpvec):
		ssum = np.sqrt(reduce(lambda x, y: x + y, map(lambda t: t**2, tmpvec)))
		if ssum > 0:
			tmpvec = map(lambda t: t / ssum, tmpvec)
		embed_out.write(' '.join(map(lambda t: str(t), tmpvec)))


	idx = 0
	with open(args.embeddings, 'r') as f:
		for embedding in f:
			if idx >= len(inses):
				break
		# assert(tweets[idx].split(args.delimiter)[0] == embeddings[idx].split(args.delimiter)[0])
		# 	sys.stdout.write('\r'+str(idx))
		# 	sys.stdout.flush()
			can_time= gettime(inses[idx])
			longitude = inses[idx][0]
			latitude = inses[idx][1] 
			# print(can_time)
			# print(cur_time)
			# print(can_time-cur_time).total_seconds()
			# tt = raw_input('one')
			if (can_time - cur_time).total_seconds() > inverval_sec:
				length_map[find] = cur_length
				find += 1
				embed_out.close()
				embed_out = open(args.output_embedding_folder+str(find)+'.embed', 'w')
				tweet_out.close()
				tweet_out = open(args.output_tweets_folder+str(find)+'.tweet', 'w')
				# cur_time = cur_time + datetime.timedelta(seconds =inverval_sec)
				cur_time = can_time
				cur_length = 0
			cur_length += 1
			tweet_out.write(tweets[idx])
			fields = embedding.strip().split('\x01')
			spatial_feature, temporal_feature, textual_feature = [ast.literal_eval(field) for field in fields[1:]]
			output_norm_vec(textual_feature)
			embed_out.write(' '+str(longitude)+' '+str(latitude)+'\n')
			idx += 1
	
	embed_len = len(textual_feature)
	
	embed_out.close()
	tweet_out.close()

	directory = os.path.dirname(args.output_setting)
	if not os.path.exists(directory):
		os.makedirs(directory)

	with open(args.output_setting, 'w') as f:
		f.write(str(find)+'\n')
		f.write(str(embed_len)+'\n')
		for (k, v) in length_map.iteritems():
			f.write(str(k)+' '+str(v)+'\n')
