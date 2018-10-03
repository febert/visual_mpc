BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
config = {
    'dataset_params': {'n_negative': 1},
    'save_dir': BASE_DIR + '/base_model/'
}