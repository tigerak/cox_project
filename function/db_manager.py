import numpy as np
import json
from hashlib import sha256
# Modules
from config import *
from function.utile.data_analy import get_data
from function.utile.chroma_util import (safe_parse_json, safe_parse_json_2, 
                                        fix_extra_closing_brace, safe_normalize)

class DBManager:
    def __init__(self, openai_api, chroma):
        self.openai_api = openai_api
        self.chroma = chroma
    
    ### 데이터 저장 도구 ###
    async def data_add_chromadb(self, data_path):
        # 데이터 정제 및 로컬 저장
        data = get_data(data_path)

        temp_data = {}
        for i, (k, v) in enumerate(data.items(), start=1):
            # OpenAI Embedding 획득
            embedding = await self.openai_api.get_embedding(k, OPENAI_EMBED_NAME)
            # key를 해싱하여 id 생성
            id = sha256(k.encode("utf-8")).hexdigest()

            # temp_data에 저장
            temp_data[k] = {
                "id": id,
                "content": v,
                "embedding": embedding
            }
            print(f"{i}번 데이터 저장 완료 : {k}")
            # if i == 20:
            #     break

        with open(SAVE_DIR + "/key_embed.json", "w", encoding="utf-8") as f:
            json.dump(temp_data, f, ensure_ascii=False, indent=2)

        # Local 데이터 ChromaDB에 저장
        with open(SAVE_DIR + "/key_embed.json", "r", encoding="utf-8") as f:
            temp_data = json.load(f)

        # ChromaDB에 저장: title -(sha256)-> id
        for i, (k, v) in enumerate(temp_data.items(), start=1):
            v["id"] = sha256(k.encode("utf-8")).hexdigest()
            
            self.chroma.add_data(id=str(v["id"]),
                                 title=k, 
                                 content=v["content"],
                                 embedding=v["embedding"])
            print(f"{i}번 데이터 ChromaDB 저장 완료 : {k}")
            
        with open(SAVE_DIR + "/key_embed.json", "w", encoding="utf-8") as f:
            json.dump(temp_data, f, ensure_ascii=False, indent=2)


    ### 데이터 검색 도구 ###
    # 기계적 DB 검색
    def search_db(self, query_text):
        # OpenAI Embedding 획득 
        query_embedding = self.openai_api.get_embedding(query_text, OPENAI_EMBED_NAME)
        # ChromaDB에 저장된 데이터와 유사도 검색
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=20,
            include=["documents", "metadatas", "distances"]
        )
        # 결과 출력
        output = []
        for i in range(len(result["documents"][0])):
            output.append({
                "title": result["documents"][0][i],
                "content": result["metadatas"][0][i]["content"],
                "distance": result["distances"][0][i]
            })

        # for i, v in enumerate(output):
        #     print(f"{i+1}번 결과 : {v['title']}")
        #     print(f"거리 : {v['distance']}")
        #     print(f"내용 : {v['content']}")
        #     print("-"*50)

        return output

    # AI 이용한 DB 검색
    async def ai_db_search(self, query_text, conversation_list):
        
        system_prompt = """# Identity

당신은 스마트스토어 상담 챗봇에서 사용할 쿼리 파서를 만드는 전문가입니다.
당신의 역할은 사용자 질문에서 검색에 사용할 핵심 키워드(include)와
검색에서 제외해야 할 키워드(exclude)를 추출하여 구조화된 JSON으로 반환하는 것입니다.

# Instructions

* 이전 대화의 흐름을 참고하여 user의 마지막 질문을 기준으로 "include"와 "exclude"를 판단하세요.
""".strip()

        parsing_prompt = """# Instructions

* 위 대화의 맥락을 고려하여 user의 마지막 질문을 정확히 분석하고 검색용 키워드를 추출하세요.
* 반드시 아래와 같은 JSON 형식으로 답변해야 합니다. 다른 텍스트는 절대 포함하지 마세요
  형식: {"include": [...], "exclude": [...]}
* "include"에는 대화의 맥락 상 사용자가 알고 싶은 핵심 키워드만를 추출하여 입력하세요. ("exclude"의 대상은 넣지 마세요.)
* "exclude"에는 '제외', '빼고', '말고' 등의 대상이 되는 키워드만 넣으세요.
제외 키워드가 없으면 "exclude": [] 로 정확히 입력하세요.
* 답변은 JSON 하나로만 출력하세요.

# Answer Guidelines

* "더 알아야 할 내용이 있나요", "그 조건이 맞지 않으면?" 처럼, 주어가 없어나 대명사라서, 대화의 맥락에서 키워드를 추출해야하는 질문이 들어온다면, 과거 대화로부터 키워드를 찾아내세요.
* user의 질문이 너무 짧거나 완벽한 문장이 아니라면, 대화의 맥락을 고려하여 완벽한 문장으로 만들어서 키워드를 추출하세요

# Examples

<user_query>
무료 배송을 제외한 나머지 배송 방법 알려줘
</user_query>

<assistant_response>
{"include":["배송 방법"], "exclude":["무료 배송"]}
</assistant_response>

<user_query>
입점 신청 시 필요한 서류
</user_query>

<assistant_response>
{"include":["입점 신청", "서류"], "exclude":[]}
</assistant_response>

# Answer Guidelines

* 절대로 단어를 함부로 바꾸지 마세요.
""".strip()
        
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        messages.extend(conversation_list[-5:])
        messages.append({"role": "system", "content": parsing_prompt})
        messages.append({"role": "user", "content": query_text})
        # OpenAI API 호출
        assistant_reply = await self.openai_api.run_chat(
                                            messages=messages,
                                            model_name=OPENAI_MODEL_NAME
                                            )
        # print(f"AI 쿼리 파서: {assistant_reply}")

        # JSON 정제
        assistant_reply = fix_extra_closing_brace(assistant_reply)
        standardized_qurey = safe_parse_json(assistant_reply)

        _include = standardized_qurey["include"]
        include_query = " ".join(_include)
        _exclude = standardized_qurey["exclude"]
        exclude_query = " ".join(_exclude)
        
        # ChromaDB에 저장된 데이터와 유사도 검색
        output = await self.conditional_search(include=include_query, 
                                               exclude=exclude_query)
        
        conditional_query = (include_query, exclude_query)
        
        return conditional_query, output

    async def conditional_search(self, include, exclude):
        """
        include keyword에서 exclude keyword를 뺀 vector를 이용해서
        ChromaDB에서 cosin similarity .
        """
        v_pos = await self.openai_api.get_embedding(include, OPENAI_EMBED_NAME)
        v_pos = np.array(v_pos)
        # 제와 조건이 없다면 0벡터
        if exclude.strip():
            v_neg = await self.openai_api.get_embedding(exclude, OPENAI_EMBED_NAME)
            v_neg = np.array(v_neg)
            q_vec = safe_normalize(v_pos) - ALPHA * safe_normalize(v_neg)
            q_vec = safe_normalize(q_vec)
        else:
            q_vec = safe_normalize(v_pos)

        result = self.chroma.collection.query(
            query_embeddings=[q_vec],
            n_results=20,
            include=["documents", "metadatas", "distances"]
        )
        
        # 결과 출력
        output = []
        for i in range(len(result["documents"][0])):
            if result["distances"][0][i] < 2.3:
                output.append({
                    "title": result["documents"][0][i],
                    "content": result["metadatas"][0][i]["content"],
                    "distance": result["distances"][0][i]
                })

        # for i, v in enumerate(output):
        #     print(f"{i+1}번 결과 : {v['title']}")
        #     print(f"거리 : {v['distance']}")
        #     # print(f"내용 : {v['content']}")
        #     print("-"*50)

        return output
    

    # AI가 제안하는 추천 질문
    async def ai_recommend(self, context, conversation_list):
        
        system_prompt = """# Identity

당신은 스마트스토어 사장님 지원 챗봇의 추천 질문 생성 전문가입니다.
"대화의 흐름"과 "Q&A Records"를 참고하여, 다음에 user가 궁금해할 만한 질문을 예측하고 제안합니다.

# Instructions

* "Q&A Records"에 있는 내용과 "assistant의 답변"만을 바탕으로, user가 다음에 궁금해할 만한 질문을 3가지 추천하세요.
"""
        recommend_prompt = f"""# Instructions

* 아래의 "Q&A Records"를 참고해, user가 다음에 궁금해할 만한 스마트스토어 질문 3가지를 만들어주세요.
* 3가지 질문은 user가 실제로 물어볼 법한 표현을 그대로 사용해 자연스럽게 작성해주세요.
* 3가지 질문은 모두 "Q&A Records"에 근거한 내용으로만 작성해주세요. 
* 반드시 "Q&A Records"에 있는 내용만으로 3가지 질문을 만드세요. 
* 지난 대화에서 이미 답변한 질문은 3가지 질문에 포함하지 마세요.
* 출력은 다음 JSON 형식으로 반환하세요 (다른 문장 X):

  {{"recommend": ["질문1", "질문2", "질문3"]}}

  {context}
""".strip()
        
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        messages.extend(conversation_list[-3:])
        messages.append({"role": "system", "content": recommend_prompt})

        # OpenAI API 호출
        assistant_reply = await self.openai_api.run_chat(
                                            messages=messages,
                                            model_name=OPENAI_MODEL_NAME
                                            )
        assistant_reply = fix_extra_closing_brace(assistant_reply)
        standardized_qurey = safe_parse_json_2(assistant_reply)

        return standardized_qurey