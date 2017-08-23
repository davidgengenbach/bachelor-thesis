import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s // %(levelname)s - %(name)s: %(message)s"
)

def get_logger(name = ''):
    return logging.getLogger(name)

LOGGER = get_logger('default')