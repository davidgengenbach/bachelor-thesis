import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s // %(levelname)s // %(message)s",
    stream=sys.stdout
    #format="%(asctime)s // %(levelname)s - %(name)s: %(message)s"
)

DEFAULT_IGNORES = ["git", "pip"]

def get_logger(name = ''):
    return logging.getLogger(name)

LOGGER = get_logger()

for package in DEFAULT_IGNORES:
    logging.getLogger(package).setLevel(logging.WARNING)