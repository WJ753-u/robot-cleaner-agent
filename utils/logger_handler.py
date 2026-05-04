import logging
from utils.path_tool import get_abs_path
import os
from utils.path_tool import get_project_root
from datetime import datetime
LOG_ROOT = get_abs_path('log')#日志保存的根目录
os.makedirs(LOG_ROOT, exist_ok=True)#创建日志根目录，如果存在则不创建
#时间，文件名，日志级别，模块名，行号，日志消息
DEFAULT_LOG_FORMAT = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')#默认的日志格式

def get_logger(
    name:str = "agent",
    console_level:int = logging.INFO,
    file_level:int = logging.DEBUG,
    log_file:str = None,
) -> logging.Logger:
    logger = logging.getLogger(name)#创建日志记录器
    logger.setLevel(console_level)#设置控制台日志级别
    if logger.handlers:#如果日志记录器已经存在处理器，直接返回。避免重复添加logger处理器
        return logger
    console_handler = logging.StreamHandler()#创建控制台日志处理器
    console_handler.setLevel(console_level)#设置控制台日志级别
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)#设置控制台日志格式
    logger.addHandler(console_handler)#添加控制台日志处理器
    if not log_file:#日志文件存放路径
        log_file = os.path.join(LOG_ROOT, f'{name}_{datetime.now().strftime("%Y%m%d%H%M%S")}.log')#默认的日志文件名
    file_handler = logging.FileHandler(log_file, encoding='utf-8')#创建日志文件处理器
    file_handler.setLevel(file_level)#设置日志文件级别
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)#设置日志文件格式
    logger.addHandler(file_handler)#添加日志文件处理器
    return logger
#快捷获取日志管理器
logger = get_logger()#直接引用logger就可以使用日志管理器
if __name__ == '__main__':
    logger.info('这是一条info日志')
    logger.debug('这是一条debug日志')
    logger.error('这是一条error日志')
    logger.warning('这是一条warning日志')
    logger.critical('这是一条critical日志')