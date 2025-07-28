import logging
import logging.handlers
import argparse
import options
import os

opt = options.Options().init(argparse.ArgumentParser(description='ShadowRemoval')).parse_args()


class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        super().__init__(fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return super().format(record)


def setup_logging(log_path=opt.log_dir):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())

    handlers = [console_handler]
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=2 * 1024 ** 2,  # 2MB
            backupCount=3
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers
    )
    return log_path