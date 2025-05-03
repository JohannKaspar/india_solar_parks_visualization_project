"""\ This module is a wrapper over the OpenAI API for the Language Model. It provides a simple interface to interact with the OpenAI API for the Language Model."""

import json
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from fastapi import HTTPException
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model, init_llm
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.native.openai import chat
from .config import settings

proxy_client = get_proxy_client("gen-ai-hub")


class LLM:
    llm = init_llm(settings.LLM_MODEL, settings.LLM_TEMP)
    embedding = init_embedding_model(settings.EMBEDDING_MODEL)
    chatOpenAI = ChatOpenAI(
        proxy_model_name=settings.LLM_MODEL, proxy_client=proxy_client
    )

    def get_completion(self, prompt):
        """Get completion from OpenAI GPT models"""
        response = None
        try:
            if self.is_messages_json(prompt):
                messages = json.loads(prompt)
                kwargs = dict(model_name=settings.LLM_MODEL, messages=messages)
            else:
                messages = [{"role": "user", "content": prompt}]
                kwargs = dict(model_name=settings.LLM_MODEL, messages=messages)

            response = chat.completions.create(**kwargs)
        except Exception as e:
            print(f"Error: {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error")

        return response.choices[0].message.content

    def get_embedding(self, input) -> str:
        """Get embeddings from OpenAI GPT models"""
        response = self.embedding.embed_query(input)
        return response

    def is_messages_json(self, prompt_text):
        try:
            messages = json.loads(prompt_text)
            if isinstance(messages, list) and all(
                isinstance(message, dict) for message in messages
            ):
                required_keys = ["role", "content"]
                if all(
                    all(key in message for key in required_keys) for message in messages
                ):
                    return True
        except json.JSONDecodeError:
            pass
        return False
    
if __name__ == "__main__":
    llm = LLM()
    prompt = "What is the capital of France?"
    response = llm.get_completion(prompt)
    print(response)
