import sys
import io
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

from time import time
import numpy as np
import re
import json
from hashlib import sha256
import asyncio
# modules
from config import *
from function.utile.data_analy import data_analysis, get_data
from function.utile.openai_util import OpenAIChat
from function.utile.chroma_util import ChromaDB


### Setting ###
# OPENAI_API_KEY = r""
# DATA_PATH = r""

class SmartAssistant:
    def __init__(self):
        # OpenAI API 초기화
        self.openai_api = OpenAIChat(OPENAI_API_KEY)
        print("OpenAI API 초기화 완료!")
        # ChromaDB 초기화
        self.chroma = ChromaDB()
        print("ChromaDB 초기화 완료!")
        
    def analysis(self):
        ### 데이터 분석 ###
        data_analysis(DATA_PATH)

    def data_add_chromadb(self):
        ### 데이터 정제 및 로컬 저장 ###
        data = get_data(DATA_PATH)

        # 데이터 저장 Sequence
        temp_data = {}
        for i, (k, v) in enumerate(data.items(), start=1):
            # OpenAI Embedding 획득 (title -> embedding)
            embedding = self.openai_api.get_embedding(k, OPENAI_EMBED_NAME)
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

        with open("./smart/data/key_embed.json", "w", encoding="utf-8") as f:
            json.dump(temp_data, f, ensure_ascii=False, indent=2)

        ### Local 데이터 ChromaDB에 저장 ###
        with open("./smart/data/key_embed.json", "r", encoding="utf-8") as f:
            temp_data = json.load(f)

        # ChromaDB에 저장: title -(sha256)-> id
        for i, (k, v) in enumerate(temp_data.items(), start=1):
            v["id"] = sha256(k.encode("utf-8")).hexdigest()
            
            self.chroma.add_data(id=str(v["id"]),
                            title=k, 
                            content=v["content"],
                            embedding=v["embedding"])
            print(f"{i}번 데이터 ChromaDB 저장 완료 : {k}")
            
        with open("./smart/data/temp_data.json", "w", encoding="utf-8") as f:
            json.dump(temp_data, f, ensure_ascii=False, indent=2)

    def search_db(self, query_text):
        ### 데이터 검색 ###
        # OpenAI Embedding 획득 (query_text -> embedding)
        query_embedding = self.openai_api.get_embedding(query_text, OPENAI_EMBED_NAME)
        # ChromaDB에 저장된 데이터와 유사도 검색
        result = self.chroma.collection.query(
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

        for i, v in enumerate(output):
            print(f"{i+1}번 결과 : {v['title']}")
            print(f"거리 : {v['distance']}")
            print(f"내용 : {v['content']}")
            print("-"*50)

        return output

    async def chat(self):
        
        conversation_list = []
        system_prompt = """# Identity

당신은 스마트스토어 고객지원 상담 챗봇입니다.

# Instructions

* 상담사는 user의 질문에 대해 친절하고 정확하게 답변해야 합니다.
* 모든 메시지를 지속적인 대화의 일부로 간주하세요.
* user가 질문한 내용에 초점을 맞춰 답변을 유지하세요.
* 환각이나 꾸며낸 정보는 포함하지 마세요.

# Example Behavior

If the user says: "배송비가 얼마인가요?"  
→ You might respond: "기본 배송비는 3,000원입니다. 단, 제주 및 도서산간 지역은 추가 요금이 부과될 수 있습니다."  
""".strip()

        # 사용자 입력 루프
        print("상담 시작! 'exit' 입력 시 종료됩니다.\n")
        while True:
            # 사용자 질문 입력
            try:
                print("사용자: ", end='', flush=True)
                raw_input = sys.stdin.buffer.readline()
                user_input = self.openai_api.embed_clean_text(raw_input)
                start_time = time()
            except Exception as e:
                print(f"입력 오류: {e}")
                continue
            # 종료 조건
            if user_input.lower().strip() == 'exit':
                print("상담 종료.")
                break

            # 관련 chunk 검색
            ###
            # rag_results = self.search_db(user_input)
            conditional_query, rag_results = await self.ai_db_search(user_input, conversation_list)
            ###
            print(f"검색 포함 키워드: {conditional_query[0]}")
            print(f"검색 제외 키워드: {conditional_query[1]}")

            if len(rag_results) == 0:
                print("저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.")
                continue
            else:
                context = "\n\n".join(
                    [f"[{i+1}번] Question: {r['title']}:\nAnswer: {r['content']}" for i, r in enumerate(rag_results)]
                )
            
            rag_prompt = f"""# Instructions

다음은 다른 스마트스토어 고객들이 남긴 Q&A Records 입니다.  
user의 마지막 질문에 대해 Q&A Records의 내용을 참고하여 답변하세요.

# Context: Q&A Records

{context}

# Answer Guidelines

* user의 마지막 질문에 대해 Q&A Records의 내용만 참고하여 답변하세요.
* 항목이 여러 개인 경우 번호를 붙여 나열하세요.
* 필요 시 구분된 소제목(예: 배송 방법, 신청 서류)으로 나누어도 좋습니다.
* "Q&A Records의 내용"과 "user의 마지막 질문"이 서로 관련 없어서서 답변이 불가능한 경우, 
  "<system_message> 관련 내용이 없어 답변할 수 없습니다."라고만 답변하세요.
* Q&A Records에 없는 정보를 만들어내거나 가정하지 마세요.

# Output Format

* 답변에 참고한 Q&A Records가 있다면, 그 번호를 답변 마지막에 반드시 기입하세요. 
  Format: "<system_message> [1, 3, 5]'

# Examples

<user_query>
사장님 보험에 대해 알려줘줘
</user_query>

<assistant_response>
사장님 보험에 대한 내용은 다음과 같습니다.

1. **보험 종류**: 사장님 보험은 크게 '4대보험/의무보험/일반보험' 세 가지 유형으로 나눌 수 있습니다.
   - 4대보험/의무보험: 법적으로 사업의 규모와 업종에 따라 필수 가입해야 하는 보험입니다.
   - 일반보험: 필요할 경우 선택적으로 가입할 수 있는 보험입니다.

2. **가입 의무**: 4대보험은 근로자를 사용하는 사업장에서는 필수로 가입해야 합니다. 미가입 시 사업주에게 법적 책임이 따를 수 있습니다.

3. **문의 방법**: 보험 상품의 보장 내용 및 가입 관련 문의는 제휴 보험사 또는 현재 이용 중인 보험사를 통해 확인하실 수 있습니다. 제휴 보험사로는 현대해상, KB손해보험, 삼성화재, 흥국화재 등이 있습니다.

자세한 사항은 보험사에 직접 문의하시기 바랍니다.
<system_message> [4, 2]
</assistant_response>

<user_query>
네이버에서 사진 찍기 좋은 장소 알려줘줘 
</user_query>

<assistant_response>
<system_message> 관련 내용이 없어 답변할 수 없습니다.
</assistant_response>
""".strip()
            # print(rag_prompt)
            # 시스템 프롬프트와 지난 대화 기록 추가
            messages = []
            messages.append({"role": "system", "content": system_prompt})
            messages.extend(conversation_list[-20:])
            messages.append({"role": "system", "content": rag_prompt})
            messages.append({"role": "user", "content": user_input})
            
            # OpenAI API 호출
            assistant_reply = await self.openai_api.run_chat(
                                                messages=messages,
                                                model_name=OPENAI_MODEL_NAME
                                                )
            # print(f"AI 상담사: {assistant_reply}")

            # 답변 파싱
            # Q&A 검색 결과가 없는 경우 처리리
            if "답변할 수 없습니다" in assistant_reply:
                print("저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.")
                
            else: # 답변에 참고한 Q&A가 있는 경우
                match = re.search(r"\s*\[([\d,\s]+)\]", assistant_reply)
                if not match: # 하지만 답변에 적합하지 않은 경우
                    print("답변에 참고할 Q&A가 없습니다. 상담사에게 연결해드릴까요?")
                    
                else: # 적합한 답변을 생성한 경우 - 연관 질문 추천
                    # 참고 Q&A 번호 추출
                    refer_numbers = match.group(1)
                    refer_numbers = [int(num.strip()) for num in refer_numbers.split(",")]
                    # print(f"참고한 질문 번호: {refer_numbers}")

                    # system_message 부분 제거
                    cleaned_reply = re.sub(r"<system_message>\s*\[[\d,\s]+\]", "", assistant_reply).strip()
                    print("챗봇:", cleaned_reply)

                    # 참고 질문 목록
                    print("-"*50)
                    print("<다음과 같은 Q&A를 참고했습니다.>")
                    refer_list = []
                    for i in refer_numbers:
                        refer_list.append(rag_results[int(i)-1]["title"])
                        print(f"참고 질문: {rag_results[int(i)-1]['title']}")
                    # 참고한 질문 제목으로 검색
                    refer_query = " ".join(refer_list)
                    conditional_query, recommend = await self.ai_db_search(refer_query, conversation_list)
                    # 추천 질문 목록
                    print("<다음과 같은 질문도 추천합니다.>")
                    recommend_list = []
                    for dict_item in recommend:
                        if len(recommend_list) >= 3:
                            break
                        if dict_item["title"] not in refer_list:
                            recommend_list.append(dict_item["title"])
                            print(f"추천 질문: {dict_item['title']}")
                    print("-"*50)

            # 대화 기록 업데이트
            conversation_list.append({"role": "user", "content": user_input})
            conversation_list.append({"role": "assistant", "content": assistant_reply})
            print(len(conversation_list))
            end_time = time()
            process_time = end_time - start_time
            print(f"처리 시간: {process_time:.2f}초")

    async def ai_db_search(self, query_text, conversation_list):
        ### 데이터 검색 ###
        # OpenAI 질문 규격화
        system_prompt = """# Identity

당신은 스마트스토어 상담 챗봇에서 사용할 쿼리 파서를 만드는 전문가입니다.
당신의 역할은 사용자 질문에서 검색에 사용할 핵심 키워드(include)와
검색에서 제외해야 할 키워드(exclude)를 추출하여 구조화된 JSON으로 반환하는 것입니다.

# Instructions

* 이전 대화의 흐름을 참고하되, user의 마지막 질문을 기준으로 "include"와 "exclude"를 판단하세요.
""".strip()

        parsing_prompt = """# Instructions

* 위 대화 흐름을 고려하여, user의 마지막 질문을 정확히 분석해서 검색용 키워드를 추출하세요.
* 반드시 아래와 같은 JSON 형식으로 답변해야 합니다. 다른 텍스트는 절대 포함하지 마세요
  형식: {"include": [...], "exclude": [...]}
* "include"에는 사용자가 알고 싶은 핵심 키워드를 추출하여 입력하세요. 
* "exclude"에는 '제외', '빼고', '말고' 등의 대상이 되는 키워드만 넣으세요.
  제외 키워드가 없으면 "exclude": [] 로 정확히 입력하세요.
* 답변은 JSON 하나로만 출력하세요.

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

        # OpenAI Embedding 획득 (query_text -> embedding)
        # standardized_qurey = json.loads(assistant_reply)
        assistant_reply = self.fix_extra_closing_brace(assistant_reply)
        standardized_qurey = self.safe_parse_json(assistant_reply)

        _include = standardized_qurey["include"]
        include_query = " ".join(_include)
        _exclude = standardized_qurey["exclude"]
        exclude_query = " ".join(_exclude)

        # print(f"검색 포함 키워드: {include_query}")
        # print(f"검색 제외 키워드: {exclude_query}")
        
        # ChromaDB에 저장된 데이터와 유사도 검색
        output = await self.conditional_search(include=include_query, 
                                               exclude=exclude_query)
        
        conditional_query = (include_query, exclude_query)
        return conditional_query, output

    async def conditional_search(self, include, exclude, alpha=0.8):
        v_pos = await self.openai_api.get_embedding(include, OPENAI_EMBED_NAME)
        v_pos = np.array(v_pos)
        # 제와 조건이 없다면 0벡터
        if exclude.strip():
            v_neg = await self.openai_api.get_embedding(exclude, OPENAI_EMBED_NAME)
            v_neg = np.array(v_neg)
            q_vec = v_pos/np.linalg.norm(v_pos) - alpha * v_neg/np.linalg.norm(v_neg)
            q_vec = q_vec/np.linalg.norm(q_vec)
        else:
            q_vec = v_pos/np.linalg.norm(v_pos)

        result = self.chroma.collection.query(
            query_embeddings=[q_vec],
            n_results=20,
            include=["documents", "metadatas", "distances"]
        )
        # 결과 출력
        output = []
        for i in range(len(result["documents"][0])):
            if result["distances"][0][i] < 10.3:
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
    
    def safe_parse_json(self, text):
        """
        문자열 내에서 JSON 객체를 추출하고, 파싱 가능한 가장 첫 JSON 블록만 반환.
        중복 괄호나 기타 오류가 있을 경우 정규식으로 보정 시도.
        """
        try:
            # 1차 정상 파싱 시도
            return json.loads(text)
        except json.JSONDecodeError:
            # 중복 괄호 제거 또는 문자열 내 JSON 블록만 추출
            json_like = re.search(r'\{.*\}', text, re.DOTALL)
            if json_like:
                json_str = json_like.group()
                # ✅ "include": [...] 와 "exclude": [...] 추출
                inc_match = re.search(r'"include"\s*:\s*\[[^\]]*\]', json_str)
                exc_match = re.search(r'"exclude"\s*:\s*\[[^\]]*\]', json_str)
                if inc_match and exc_match:
                    recovered = '{' + inc_match.group() + ', ' + exc_match.group() + '}'
                    try:
                        return json.loads(recovered)
                    except json.JSONDecodeError as e3:
                        print("🔴 복구된 문자열 파싱 실패:", e3)
                    
            raise ValueError("유효한 JSON 형식을 찾을 수 없습니다.")
        
    def fix_extra_closing_brace(self, text: str) -> str:
        open_braces = text.count('{')
        close_braces = text.count('}')
        # 닫는 중괄호가 더 많다면 초과만큼 제거
        if close_braces > open_braces:
            diff = close_braces - open_braces
            for _ in range(diff):
                # 뒤에서부터 하나씩 제거
                text = text[::-1].replace('}', '', 1)[::-1]
        return text

if __name__ == "__main__":
    smart = SmartAssistant()
    asyncio.run(smart.chat())

    # query_text = "원쁠템과 원쁠딜은 어떤 차이야?"
#     query_text = """ [원쁠템] 원쁠템과 원쁠딜 동일 기간 중복 제안 시 원쁠템 가격은 별도로 수정하지 않아도 되나요?
# [원쁠딜] 원쁠딜인가요, 원뿔딜 인가요?
# [원쁠딜] 원쁠딜은 어떤 서비스인가요?
# [원쁠딜] 원쁠딜에서 판매하고 싶으면 어떻게 하면 되나요?
# [공통] '원쁠딜/원쁠템' 최대할인가와 실구매가가 왜 다르게 보이나요?
# """
    query_text = "알리에 물건을 팔고 싶은데 어떻게 해?" # 1.3 이하
    # query_text = ""
    # asyncio.run(smart.db_test(query_text))