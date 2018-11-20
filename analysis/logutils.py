import logging
import sys


def create_logger(logger_name, log_file, log_to_stdout=True, log_level='DEBUG'):
    logger = logging.getLogger(logger_name)
    file_log_handler = logging.FileHandler(log_file)
    logger.addHandler(file_log_handler)

    if log_to_stdout:
        stdout_log_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_log_handler)

    # nice output format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_log_handler.setFormatter(formatter)
    stdout_log_handler.setFormatter(formatter)

    logger.setLevel('DEBUG')
    return logger
