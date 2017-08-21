data_dir='../../../data/toy/'
python ../split_batches.py --data_dir $data_dir
python ../gen_queries.py $data_dir
