"""
LLM Integration module for Agentic RAG
Supports Groq and OpenAI
"""
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
import os


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def generate_with_streaming(self, prompt: str, **kwargs):
        """Generate text with streaming"""
        pass


class GroqLLM(LLMProvider):
    """Groq LLM provider"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        from langchain_groq import ChatGroq
        self.client = ChatGroq(
            api_key=self.api_key,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Groq"""
        from langchain_core.messages import HumanMessage

        kwargs.setdefault("temperature", self.temperature)
        kwargs.setdefault("max_tokens", self.max_tokens)

        message = HumanMessage(content=prompt)
        response = self.client.invoke([message])
        return response.content

    def generate_with_streaming(self, prompt: str, **kwargs):
        """Generate text with streaming from Groq"""
        from langchain_core.messages import HumanMessage
        
        message = HumanMessage(content=prompt)
        for chunk in self.client.stream([message]):
            yield chunk.content


class OpenAILLM(LLMProvider):
    """OpenAI LLM provider"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        from langchain_openai import ChatOpenAI
        self.client = ChatOpenAI(
            api_key=self.api_key,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI"""
        from langchain_core.messages import HumanMessage

        kwargs.setdefault("temperature", self.temperature)
        kwargs.setdefault("max_tokens", self.max_tokens)

        message = HumanMessage(content=prompt)
        response = self.client.invoke([message])
        return response.content

    def generate_with_streaming(self, prompt: str, **kwargs):
        """Generate text with streaming from OpenAI"""
        from langchain_core.messages import HumanMessage
        
        message = HumanMessage(content=prompt)
        for chunk in self.client.stream([message]):
            yield chunk.content


def get_llm_provider(
    provider: str = "groq",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2000
) -> LLMProvider:
    """Factory function to get LLM provider"""
    
    if provider.lower() == "groq":
        if model is None:
            model = "mixtral-8x7b-32768"
        return GroqLLM(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif provider.lower() == "openai":
        if model is None:
            model = "gpt-3.5-turbo"
        return OpenAILLM(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
