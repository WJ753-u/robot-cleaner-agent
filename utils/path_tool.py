import os

def get_project_root() -> str:
    current_file = os.path.abspath(__file__)#当前文件的绝对路径
    current_dir = os.path.dirname(current_file)#获取文件所在的文件夹的绝对路径
    project_root = os.path.dirname(current_dir)#再向上一级，找到项目根目录
    return project_root
def get_abs_path(relative_path: str) -> str:
    project_root = get_project_root()#拿到过程根目录
    return os.path.join(project_root, relative_path)#整个工程根目录和相对路径，拼接起来得到完整的路径
if __name__ == '__main__':
    print(get_abs_path('config/config.txt'))