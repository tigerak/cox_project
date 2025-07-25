import os
import argparse
import asyncio
# Modules
from config import *
from function.utile.data_analy import data_analysis
from function.db_manager import DBManager
from function.chat_manager import ChatManager
from function.utile.chroma_util import ChromaDB
from function.utile.openai_util import OpenAIChat


### API Setting ###
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class SmartAssistant:
    def __init__(self):
        # OpenAI API 초기화
        self.openai_api = OpenAIChat(OPENAI_API_KEY)
        print("OpenAI API 초기화 완료!")
        # ChromaDB 초기화
        self.chroma = ChromaDB()
        print("ChromaDB 초기화 완료!")
        
        self.db_manager = DBManager(openai_api=self.openai_api,
                                    chroma=self.chroma)
        self.chat_manager = ChatManager(openai_api=self.openai_api,
                                        db_manager=self.db_manager)
        
    def analysis(self):
        """데이터 분석"""
        data_analysis(data_path=DATA_PATH)

    def add_chromadb(self):
        """데이터 정제 및 저장"""
        asyncio.run(self.db_manager.data_add_chromadb(data_path=DATA_PATH))
        
    def run_chatbot(self):
        """챗봇 실행 - CLI"""
        asyncio.run(self.chat_manager.cli_chatbot())
    
    def stream_chat(self, user_input, session_id):
        """챗봇 실행 - API"""
        return self.chat_manager.get_reply_stream(user_input=user_input, 
                                                  session_id=session_id)
                    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["db_setting", "run_chat"], 
                        required=True,
                        help="실행 모드 선택: db_setting | run_chat")
    args = parser.parse_args()

    smart = SmartAssistant()

    if args.mode == "db_setting":
        smart.add_chromadb()
    elif args.mode == "run_chat":
        smart.run_chatbot()