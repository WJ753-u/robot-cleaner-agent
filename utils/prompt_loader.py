from utils.config_handler import prompts_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def load_system_prompts() -> dict:
    try:
        system_prompt_path = get_abs_path(prompts_config["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[系统提示加载]配置文件中缺少{str(e)}")
        raise e
    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[系统提示加载]文件{system_prompt_path}读取失败：{str(e)}")
        raise e
def load_rag_prompts() -> dict:
    try:
        rag_prompt_path = get_abs_path(prompts_config["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[rag提示加载]配置文件中缺少{str(e)}")
        raise e
    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[rag提示加载]文件{rag_prompt_path}读取失败：{str(e)}")   
        raise e
def load_report_prompts() -> dict:
    try:
        report_prompt_path = get_abs_path(prompts_config["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[report提示加载]配置文件中缺少{str(e)}")
        raise e
    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[report提示加载]文件{report_prompt_path}读取失败：{str(e)}")   
        raise e
if __name__ == "__main__":
    # print(load_system_prompts())
    # print(load_rag_prompts())
    print(load_report_prompts())