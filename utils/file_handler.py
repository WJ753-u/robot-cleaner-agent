import os
import hashlib
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)


def get_file_md5_hex(file_path:str) -> str:#获取文件的md5值的十六进制字符串，来对文件去重
    if not os.path.exists(file_path):
        logger.error(f"[md5计算]文件{file_path}不存在")
        return None
    if not os.path.isfile(file_path):
        logger.error(f"[md5计算]文件{file_path}不是文件")
        return None
    md5_obj = hashlib.md5()
    chunk_size = 4096#4kb分片，避免内存溢出
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):#:=赋值运算符，将f.read(chunk_size)赋值给chunk,先拿到chunk_size赋值给chunk再进行while判断
                md5_obj.update(chunk)#更新md5对象
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"[md5计算]文件{file_path}读取失败：{str(e)}")
        return None
    
def listdir_with_allowed_type(path:str, allowed_types:tuple[str]):#返回文件夹内的文件列表（允许的文件后缀）,给定允许的文件类型
    files = []
    if not os.path.isdir(path):
        logger.error(f"[文件列表]路径{path}不是文件夹")
        return allowed_types
    for f in os.listdir(path):#列出文件夹内的所有文件
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))#将文件夹路径和文件名拼接起来，一起返回
    return tuple(files)#返回元组，避免被修改
def pdf_loader(filepath:str, password:str = None) ->list[Document]:
    return PyPDFLoader(filepath, password=password).load()#.load()方法返回一个Document对象列表，全量加载
def txt_loader(filepath:str) ->list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()#.load()方法返回一个Document对象列表，全量加载
def doc_loader(filepath:str) ->list[Document]:
    return UnstructuredWordDocumentLoader(filepath, mode="elements").load()
def docx_loader(filepath:str) ->list[Document]:
    return Docx2txtLoader(filepath).load()
def excel_loader(filepath:str) ->list[Document]:
    return UnstructuredExcelLoader(filepath, mode="elements").load()
def pptx_loader(filepath:str) ->list[Document]:
    return UnstructuredPowerPointLoader(filepath, mode="elements").load()
def csv_loader(filepath:str) ->list[Document]:
    return CSVLoader(filepath, encoding="utf-8-sig").load()
