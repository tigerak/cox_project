import numpy as np
import openai
import unicodedata

class OpenAIChat:
    def __init__(self, api_key):
        self.client = openai.AsyncClient(api_key=api_key)

    async def run_chat(self, messages, model_name):
        response = await self.client.chat.completions.create(
            messages=messages,
            model=model_name,
        )
        return response.choices[0].message.content
    
    async def stream_chat(self, messages, model_name):
        stream = await self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    
    async def get_embedding(self, text, model_name):
        cleaned_text = self.embed_clean_text(text)

        if not cleaned_text.strip():
            return np.zeros(1536, dtype=float).tolist()
    
        response = await self.client.embeddings.create(
            input=[cleaned_text],
            model=model_name
        )
        # print(f"임베팅 추출 완료 : {cleaned_text}")
        return response.data[0].embedding
        
    
    def clean_text(self, text):
        cleaned_text = unicodedata.normalize("NFKD", text).encode("utf-8", "ignore").decode("utf-8")
        return cleaned_text

    def embed_clean_text(self, raw_input):
        if isinstance(raw_input, bytes):
            try:
                user_input = raw_input.decode("utf-8")
            except UnicodeDecodeError:
                # user_input = raw_input.decode("utf-8", "ignore")
                user_input = raw_input.decode("euc-kr")
        else:
            user_input = raw_input
        cleaned_text = unicodedata.normalize("NFC", user_input)
        return cleaned_text