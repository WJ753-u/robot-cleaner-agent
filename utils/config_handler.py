from utils.path_tool import get_abs_path
import yaml
def load_rag_config(config_path:str = 'config/rag.yml', encoding:str = 'utf-8'):#加载rag配置文件
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_chroma_config(config_path:str = 'config/chroma.yml', encoding:str = 'utf-8'):#加载chroma配置文件
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_prompts_config(config_path:str = 'config/prompts.yml', encoding:str = 'utf-8'):#加载prompts配置文件   
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_agent_config(config_path:str = 'config/agent.yml', encoding:str = 'utf-8'):#加载agent配置文件 
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

rag_config = load_rag_config()
chroma_config = load_chroma_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()

if __name__ == '__main__':
    print(agent_config['chat_model_name'])