from .ana import ana_helper

def fetch():
    return ana_helper.get_ana_set('webkb', preprocessing_type = 'stemmed')
