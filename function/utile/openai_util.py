import openai
import unicodedata

class OpenAIChat:
    def __init__(self, api_key):
        self.client = openai.AsyncClient(api_key=api_key)

    async def run_chat(self, messages, model_name):
        response = await self.client.chat.completions.create(
            messages=messages,
            model=model_name,
            # temperature=0.7
        )
        return response.choices[0].message.content
    
    async def get_embedding(self, text, model_name):
        # cleaned_text = self.embed_clean_text(text)
        cleaned_text = text
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
        try:
            user_input = raw_input.decode("utf-8")
        except UnicodeDecodeError:
            # user_input = raw_input.decode("utf-8", "ignore")
            user_input = raw_input.decode("euc-kr")
        cleaned_text = unicodedata.normalize("NFC", user_input)
        return cleaned_text