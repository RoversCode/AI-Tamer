import logging
import os
from utils import cmm_logger_format

cmm_logger_format.init()


class Logger:
    # NOTSET < DEBUG < INFO < WARNING < ERROR < CRITICAL
    def __init__(self, logger_name=None):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        # self.fmt = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] [%(funcName)s %(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S')
        self.fmt = cmm_logger_format.JsonFormatter()

    def get_logger(self, logger_file=None, level=logging.INFO):
        if logger_file:
            dirname = os.path.dirname(logger_file)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            # 设置文件日志
            fh = logging.FileHandler(logger_file, encoding='utf-8')
            fh.setFormatter(self.fmt)
            fh.setLevel(level)
            self.logger.addHandler(fh)

        return self.logger


if __name__ == '__main__':
    logyyx = Logger('log')
    logger = logyyx.get_logger('../output/logs/test.log')
    logger.info('111')
    logger.warning('1111')
    logger.error('11111111')
