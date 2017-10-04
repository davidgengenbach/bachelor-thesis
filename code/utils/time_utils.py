import datetime

def get_time_formatted():
    return str(datetime.datetime.now())

def get_timestamp():
    return datetime.datetime.now().timestamp()

def seconds_to_human_readable(s):
    return str(datetime.timedelta(seconds=s))