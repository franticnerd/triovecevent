data_dir='../../data/toy/'
cd ..
python './gen_candidate_json.py' $data_dir
python './gen_candidate_features.py' $data_dir
# python '../gen_event_labels.py' $data_dir
# python '../eval_event_classification.py' $data_dir

