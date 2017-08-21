import numpy as np
import random
from sklearn.cluster import MeanShift
from collections import defaultdict
import time, datetime
from time import time as cur_time
import itertools
import sys, os, ast
from subprocess import call, check_call
from scipy.special import expit
import math
from sklearn.neighbors import NearestNeighbors
from copy import deepcopy
from shutil import copyfile, rmtree
from sklearn.metrics.pairwise import cosine_similarity

class CrossMap(object):
	def __init__(self, pd):
		self.pd = pd
		self.lClus = LMeanshiftClus(pd)
		self.tClus = TMeanshiftClus(pd)
		self.nt2vecs = None # center vectors
		self.nt2cvecs = None # context vectors
		self.embed_algo = GraphEmbed(pd)

	def fit(self, tweets, voca):
		random.shuffle(tweets)
		locations = [[tweet.lat, tweet.lng] for tweet in tweets]
		times = [[tweet.ts] for tweet in tweets]
		self.lClus.fit(locations)
		print 'spatial cluster num:', len(self.lClus.centroids)
		self.tClus.fit(times)
		print 'temporal cluster num:', len(self.tClus.centroids)
		nt2nodes, et2net = self.prepare_training_data(tweets, voca) # nt stands for "node type", and et stands for "edge type"
		sample_size = len(tweets)*self.pd['epoch']
		self.nt2vecs, self.nt2cvecs = self.embed_algo.fit(nt2nodes, et2net, sample_size)

	def prepare_training_data(self, tweets, voca):
		nt2nodes = {nt:set() for nt in self.pd['nt_list']}
		et2net = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
		for tweet in tweets:
			l = self.lClus.predict([tweet.lat, tweet.lng])
			t = self.tClus.predict([tweet.ts])
			c = tweet.category
			words = [w for w in tweet.words if w in voca] # from text, only retain those words appearing in voca
			nts = self.pd['nt_list'][1:]
			if 'c' in nts and c not in self.pd['category_list']:
				nts.remove('c')
			for nt1 in nts:
				nt2nodes[nt1].add(eval(nt1))
				for nt2 in nts:
					if nt1!=nt2:
						et2net[nt1+nt2][eval(nt1)][eval(nt2)] += 1
			for w in words:
				nt1 = 'w'
				nt2nodes[nt1].add(eval(nt1))
				for nt2 in nts:
					et2net[nt1+nt2][eval(nt1)][eval(nt2)] += 1
					et2net[nt2+nt1][eval(nt2)][eval(nt1)] += 1
			for w1, w2 in itertools.combinations(words, r=2):
				if w1!=w2:
					et2net['ww'][w1][w2] += 1
					et2net['ww'][w2][w1] += 1
		for nt, clus in [('l',self.lClus), ('t',self.tClus)]:
			if type(clus) == MeanshiftClus:
				'''
				strangely, MeanshiftClus seems to produce some empty clusters, but we still have to 
				include them for encoding continuous proximity and for making predictions on test data
				'''
				for cluster in range(len(clus.centroids)):
					nt2nodes[nt].add(cluster)
			self.encode_continuous_proximity(nt, clus, et2net, nt2nodes)
		return nt2nodes, et2net

	def encode_continuous_proximity(self, nt, clus, et2net, nt2nodes):
		et = nt+nt
		nodes = nt2nodes[nt]
		for n1 in nodes:
			center = clus.centroids[n1]
			for n2, proximity in clus.get_top_nbs(center):
				if n1!=n2:
					et2net[et][n1][n2] = proximity
					et2net[et][n2][n1] = proximity

	def get_nt2vecs(self, is_predict_type):
		if not is_predict_type and self.pd['second_order'] and self.pd['use_context_vec']:
			return self.nt2cvecs
		else:
			return self.nt2vecs

	def get_kernel_smoothed_vec(self, top_nbs, node2vec):
		top_nbs = [(node,weight) for node, weight in top_nbs if node in node2vec]
		if not top_nbs:
			vec = np.zeros(self.pd["dim"])
		elif top_nbs[0][1]==0:
			node = top_nbs[0][0]
			vec = node2vec[node]
		else:
			vec = np.average([node2vec[node]*weight for node, weight in top_nbs], axis=0)
		return vec

	def gen_spatial_feature(self, lat, lng, predict_type):
		nt2vecs = self.get_nt2vecs('l'==predict_type)
		location = [lat, lng]
		top_nbs = self.lClus.get_top_nbs(location)
		return self.get_kernel_smoothed_vec(top_nbs, nt2vecs['l'])

	def gen_temporal_feature(self, time, predict_type):
		nt2vecs = self.get_nt2vecs('t'==predict_type)
		time = [time]
		top_nbs = self.tClus.get_top_nbs(time)
		return self.get_kernel_smoothed_vec(top_nbs, nt2vecs['t'])

	def gen_textual_feature(self, words, predict_type):
		nt2vecs = self.get_nt2vecs('w'==predict_type)
		w_vecs = [nt2vecs['w'][w] for w in words if w in nt2vecs['w']]
		ws_vec = np.average(w_vecs, axis=0) if w_vecs else np.zeros(self.pd['dim'])
		return ws_vec

	def gen_category_feature(self, c, predict_type):
		nt2vecs = self.get_nt2vecs('c'==predict_type)
		c_vec = nt2vecs['c'][c] if c in nt2vecs['c'] else np.zeros(self.pd['dim'])
		return c_vec

	def predict(self, time, lat, lng, words, category, predict_type):
		# if 'c' not in self.pd['nt_list']:
		# 	words += category.lower().split()
		l_vec = self.gen_spatial_feature(lat, lng, predict_type)
		t_vec = self.gen_temporal_feature(time, predict_type)
		w_vec = self.gen_textual_feature(words, predict_type)
		vecs = [l_vec, t_vec, w_vec]
		if 'c' in self.pd['nt_list']:
			c_vec = self.gen_category_feature(category, predict_type)
			vecs.append(c_vec)
		score = sum([self.cosine(vec1, vec2) for vec1, vec2 in itertools.combinations(vecs, r=2)])
		return round(score, 6)

	def get_vec(self, query):
		nt2vecs = self.nt2vecs
		# use the "Python type" of the query to determine the "node type" of the query 
		if type(query)==str:
			if query in self.pd['category_list']:
				return nt2vecs['c'][query], 'c'
			else:
				return nt2vecs['w'][query.lower()], 'w'
		elif type(query)==list:
			return nt2vecs['l'][self.lClus.predict(query)], 'l'
		else:
			return nt2vecs['t'][self.tClus.predict(query)], 't'

	def get_nbs1(self, query, nb_nt, neighbor_num=20):
		vec_query, query_nt = self.get_vec(query)
		nb2vec = self.get_nt2vecs(nb_nt==query_nt)[nb_nt]
		nbs = sorted(nb2vec, key=lambda nb:self.cosine(nb2vec[nb], vec_query), reverse=True)
		nbs = nbs[:neighbor_num]
		if nb_nt=='l':
			nbs = [self.lClus.centroids[nb] for nb in nbs]
		if nb_nt=='t':
			nbs = [time.strftime('%H:%M:%S', time.gmtime(self.tClus.centroids[nb])) for nb in nbs]
		return nbs

	def get_nbs2(self, query1, query2, func, nb_nt, neighbor_num=20):
		vec_query1, query1_nt = self.get_vec(query1)
		vec_query2, query2_nt = self.get_vec(query2)
		nb2vec = self.get_nt2vecs(nb_nt in [query1_nt,query2_nt])[nb_nt]
		nbs = sorted(nb2vec, key=lambda nb:func(self.cosine(nb2vec[nb], vec_query1), self.cosine(nb2vec[nb], vec_query2)), reverse=True)
		nbs = nbs[:neighbor_num]
		if nb_nt=='l':
			nbs = [self.lClus.centroids[nb] for nb in nbs]
		if nb_nt=='t':
			nbs = [time.strftime('%H:%M:%S', time.gmtime(self.tClus.centroids[nb])) for nb in nbs]
		return nbs

	def cosine(self, list1, list2):
		return cosine_similarity([list1],[list2])[0][0]

class Clus(object):
	def __init__(self, pd):
		self.nbrs = NearestNeighbors(n_neighbors=pd['kernel_nb_num'])
		self.centroids = None

	def fit(self, X):
		pass

	def predict(self, x):
		pass

	def get_top_nbs(self, x):
		[distances], [indices] = self.nbrs.kneighbors([x])
		return [(index, self.kernel(distance, self.kernel_bandwidth)) for index, distance in zip(indices, distances)]

	def kernel(self, u, h=1.0):
		u /= h
		return 0 if u>1 else math.e**(-u*u/2)

class LMeanshiftClus(object):
	def __new__(cls, pd):
		return MeanshiftClus(pd, pd["bandwidth_l"], pd["kernel_bandwidth_l"])

class TMeanshiftClus(object):
	def __new__(cls, pd):
		return MeanshiftClus(pd, pd["bandwidth_t"], pd["kernel_bandwidth_t"])

class MeanshiftClus(Clus):
	def __init__(self, pd, bandwidth, kernel_bandwidth):
		super(MeanshiftClus, self).__init__(pd)
		self.kernel_bandwidth = kernel_bandwidth
		self.ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=10)

	def fit(self, X):
		X = np.array(X)
		self.ms.fit(X)
		self.centroids = self.ms.cluster_centers_
		self.nbrs.fit(self.centroids)

	def predict(self, x):
		return self.ms.predict([x])[0]

class GraphEmbed(object):
	def __init__(self, pd):
		self.pd = pd
		self.nt2vecs = dict()
		self.nt2cvecs = dict()
		self.path_prefix = 'GraphEmbed/'
		self.path_suffix = '-'+str(os.getpid())+'.txt'

	def fit(self, nt2nodes, et2net, sample_size):
		self.write_line_input(nt2nodes, et2net)
		self.execute_line(sample_size)
		self.read_line_output()
		return self.nt2vecs, self.nt2cvecs

	def write_line_input(self, nt2nodes, et2net):
		if 'c' not in nt2nodes: # add 'c' nodes (with no connected edges) to comply to Line's interface
			nt2nodes['c'] = self.pd['category_list']
		for nt, nodes in nt2nodes.items():
			# print nt, len(nodes)
			node_file = open(self.path_prefix+'node-'+nt+self.path_suffix, 'w')
			for node in nodes:
				node_file.write(str(node)+'\n')
		all_et = [nt1+nt2 for nt1, nt2 in itertools.product(nt2nodes.keys(), repeat=2)]
		for et in all_et:
			edge_file = open(self.path_prefix+'edge-'+et+self.path_suffix, 'w')
			if et in et2net:
				for u, u_nb in et2net[et].items():
					for v, weight in u_nb.items():
						edge_file.write('\t'.join([str(u), str(v), str(weight), 'e'])+'\n')

	def execute_line(self, sample_size):
		command = ['./hin2vec']
		command += ['-size', str(self.pd['dim'])]
		command += ['-negative', str(self.pd['negative'])]
		command += ['-alpha', str(self.pd['alpha'])]
		sample_num_in_million = max(1, sample_size/1000000)
		command += ['-samples', str(sample_num_in_million)]
		command += ['-threads', str(10)]
		command += ['-second_order', str(self.pd['second_order'])]
		command += ['-job_id', str(os.getpid())]
		# call(command, cwd=self.path_prefix, stdout=open('stdout.txt','wb'))
		call(command, cwd=self.path_prefix)

	def read_line_output(self):
		for nt in self.pd['nt_list']:
			for nt2vecs,vec_type in [(self.nt2vecs,'output-'), (self.nt2cvecs,'context-')]:
				vecs_path = self.path_prefix+vec_type+nt+self.path_suffix
				vecs_file = open(vecs_path, 'r')
				vecs = dict()
				for line in vecs_file:
					node, vec_str = line.strip().split('\t')
					try:
						node = ast.literal_eval(node)
					except: # when nt is 'w', the type of node is string
						pass
					vecs[node] = np.array([float(i) for i in vec_str.split(' ')])
				nt2vecs[nt] = vecs
		for f in os.listdir(self.path_prefix): # clean up the tmp files created by this execution
		    if f.endswith(self.path_suffix):
		        os.remove(self.path_prefix+f)
