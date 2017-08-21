def load_toy_params():
    pd = dict()
    pd['raw_data_dir'] = '/Users/chao/data/source/tweets-10k/clean/'
    pd['data_dir'] = '../../data/toy/'
    pd['pos_types'] = ['N', '^', 'S', 'Z', '#', 'B']
    pd['freq_thre'] = 100
    pd['infreq_thre'] = 5
    return pd


def load_la_params():
    pd = dict()
    pd['raw_data_dir'] = '/shared/data/czhang82/source/tweets-la/clean/'
    pd['data_dir'] = '../../data/la/'
    pd['pos_types'] = ['N', '^', 'S', 'Z', '#', 'B']
    pd['freq_thre'] = 200000
    pd['infreq_thre'] = 100
    return pd


def load_ny_params():
    pd = dict()
    pd['raw_data_dir'] = '/shared/data/czhang82/source/tweets-ny/clean/'
    pd['data_dir'] = '../../data/ny/'
    pd['pos_types'] = ['N', '^', 'S', 'Z', '#', 'B']
    pd['freq_thre'] = 200000
    pd['infreq_thre'] = 100
    return pd
