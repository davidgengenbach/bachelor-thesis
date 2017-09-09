from .ana import ana_helper

def fetch():
    return ana_helper.get_ana_set('20ng', preprocessing_type="stemmed")
