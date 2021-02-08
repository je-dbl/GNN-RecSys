"""
config_logging.py

This module aims to define the logger object.
"""
import logging


def get_logger(name):
    """
    This function aims to define the logger object from a (file) name.

    input:
            :name: str, file name
    output:
            :logger: logger object to do logging
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        # stream handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger