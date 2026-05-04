from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chat_models import init_chat_model
from utils.config_handler import rag_config
from langchain_ollama import OllamaEmbeddings

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass
class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return init_chat_model(model = rag_config["chat_model_name"], base_url = rag_config["chat_base_url"], model_provider = rag_config["model_provider"])
class EmbeddingModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return OllamaEmbeddings(model = rag_config["embedding_model_name"], base_url = rag_config["embedding_base_url"])
chat_model = ChatModelFactory().generator()#调用ChatModelFactory的generator方法，创建一个ChatModel实例
embedding_model = EmbeddingModelFactory().generator()#调用EmbeddingModelFactory的generator方法，创建一个EmbeddingModel实例
