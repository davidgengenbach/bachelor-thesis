from data.datasets.ana import ana_helper

def fetch():
    return ana_helper.get_ana_set('cade', preprocessing_type = 'stemmed')