import sys
import re
from time import time
# modules
from config import *


class ChatManager:
    def __init__(self, openai_api, db_manager):
        self.openai_api = openai_api
        self.db_manager = db_manager

        self.sessions = {}
        
    async def get_reply_stream(self, user_input, session_id="default"):
        """세션별 히스토리 유지 + GPT 토큰 스트림을 그대로 yield"""
        conversation = self.sessions.setdefault(session_id, [])
        recommend = []
        recommend_buf = []

        # RAG 검색
        condition_query, rag_results = await self.db_manager.ai_db_search(user_input, conversation)
        
        # messages 구성
        messages = self._build_messages(user_input, conversation, rag_results)

        # GPT 스트림
        assistant_buffer = ""
        printed_len = 0
        hide = False
        async for token in self.openai_api.stream_chat(messages, OPENAI_MODEL_NAME):
            assistant_buffer += token
            if hide:
                continue
            tag_pos = assistant_buffer.find(HIDDEN_TAG, printed_len)
            if tag_pos != -1:
                # 태그 직전까지 yield
                if tag_pos > printed_len:
                    yield assistant_buffer[printed_len:tag_pos]
                printed_len = len(assistant_buffer)
                hide = True
                continue
            safe_end = max(len(assistant_buffer) - LOOKAHEAD, printed_len)
            if safe_end > printed_len:
                yield assistant_buffer[printed_len:safe_end]
                printed_len = safe_end

        # 대화 기록 업데이트
        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": assistant_buffer})

        # 참고 Q&A / 추천 Q&A 
        if "답변할 수 없습니다" in assistant_buffer:
            msg = "\nQ&A에 없는 내용입니다. 상담사를 연결해 드릴까요?\n"
            yield msg
        else:
            refer_recom_list, recommend = await self._db_recom(
                                                        assistant_reply=assistant_buffer,
                                                        rag_results=rag_results,
                                                        conversation=conversation,
                                                    )
            yield ("-" * 50) + ("\n<다음과 같은 Q&A를 참고했습니다.>")
            for refer_title in refer_recom_list[0]:
                yield "\nc참고 질문:" + refer_title
            yield "\n<다음과 같은 질문도 추천합니다.>"
            for recom_title in refer_recom_list[1]:
                yield "\n추천 질문:" + recom_title
            yield "\n" + ("-" * 50)

        # AI 추천 질문
        ai_recom_dict = await self._ai_recom(
            conversation=conversation, recommend=recommend
        )
        yield "\n<AI가 추천하는 이런 질문은 어떠세요?>"
        for title in ai_recom_dict["recommend"]:
            yield "\nAI 추천 질문:" + title
        yield "\n" + ("-" * 50)


    async def cli_chatbot(self, session_id: str = "cli") -> None:
        """CLI에서도 스트리밍으로 답변을 출력한다."""
        conversation = self.sessions.setdefault(session_id, [])
        recommend = []
        recommend_buf = []
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
            condition_query, rag_results = await self.db_manager.ai_db_search(user_input, conversation)
            print("검색 포함 키워드 :", condition_query[0])
            print("검색 제외 키워드 :", condition_query[1])
            if len(rag_results) == 0:
                print("저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.")
                # AI 추천 질문 생성
                if not recommend:
                    recommend = recommend_buf
                ai_recom_dict = await self._ai_recom(conversation=conversation,
                                                     recommend=recommend)
                continue
            else:
                messages = self._build_messages(user_input, conversation, rag_results)

            # GPT 스트림 실시간 출력
            try:
                assistant_reply = await self._stream_to_buffer(
                    self.openai_api.stream_chat(messages, OPENAI_MODEL_NAME)
                )
            except Exception as e:
                print(f"API 호출 오류: {e}")
                print("API 호출에 실패했습니다. 잠시 후 다시 시도해주세요.")
                continue

            # 대화 기록 업데이트
            conversation.append({"role": "user", "content": user_input})
            conversation.append({"role": "assistant", "content": assistant_reply})
            print(f"(처리 시간 {time() - start_time:.2f}s)")

            # 질문과 매칭되는 Q&A 검색 결과가 없는 경우 처리
            recommend = []
            if "답변할 수 없습니다" in assistant_reply:
                print("죄송합니다. Q&A에 없는 내용입니다. 상담사를 연결해 드릴까요?")
            else: # 답변할 수 있다면 참고 Q&A와 추천 질문 생성성
                _, recommend = await self._db_recom(assistant_reply=assistant_reply,
                                                    rag_results=rag_results,
                                                    conversation=conversation)
                print(f"(처리 시간 {time() - start_time:.2f}s)")
            # AI 추천 질문 생성
            if not recommend:
                recommend = recommend_buf
            ai_recom_dict = await self._ai_recom(conversation=conversation,
                                                 recommend=recommend)
            print(f"(처리 시간 {time() - start_time:.2f}s)")
            # 대화 기록 업데이트
            conversation.append({"role": "assistant", "content": f"AI가 추천하는 이런 질문은 어떠세요? : {ai_recom_dict['recommend']}"})
            if recommend:
                recommend_buf = recommend


    async def _stream_to_buffer(self, agen):
        """
        HIDDEN_TAG가 나타날 때까지 토큰 스트림을 CLI에 실시간 출력하고 
        전체 텍스트를 반환.
        """
        buffer = ""
        printed_len = 0
        hide = False
        async for token in agen:
            buffer += token
            if hide:
                continue 

            # 1) 버퍼에서 태그 위치 탐색
            tag_pos = buffer.find(HIDDEN_TAG, printed_len)
            if tag_pos != -1:
                # 태그 직전까지 출력, 이후부터 숨김
                print(buffer[printed_len:tag_pos], end="", flush=True)
                printed_len = len(buffer)  # 태그 포함 위치까지로 갱신
                hide = True
                continue

            # 2) look‑ahead 를 두고 안전 구간까지만 출력
            safe_end = max(len(buffer) - LOOKAHEAD, printed_len)
            if safe_end > printed_len:
                print(buffer[printed_len:safe_end], end="", flush=True)
                printed_len = safe_end

        # 스트림 종료 후 태그가 없었다면 잔여 부분 flush
        if not hide and printed_len < len(buffer):
            print(buffer[printed_len:], end="", flush=True)
        print()  
        return buffer
    

    async def _db_recom(self, assistant_reply, rag_results, conversation):
        refer_list = []
        recommend = []
        recommend_list = []
        match = re.search(r"\s*\[([\d,\s]+)\]", assistant_reply)
        if not match: # 하지만 답변에 적합하지 않은 경우
            print("답변에 참고할 Q&A가 없습니다. 상담사에게 연결해드릴까요?")
            
        else: # 적합한 답변을 생성한 경우 - 연관 질문 추천
            # 참고 Q&A 번호 추출
            refer_numbers = match.group(1)
            refer_numbers = [int(num.strip()) for num in refer_numbers.split(",")]
            # print(f"참고한 질문 번호: {refer_numbers}")

            # 참고 질문 목록
            print("-"*50)
            print("<다음과 같은 Q&A를 참고했습니다.>")
            for i in refer_numbers:
                refer_list.append(rag_results[int(i)-1]["title"])
                print(f"참고 질문: {rag_results[int(i)-1]['title']}")

            # 참고한 질문 제목으로 검색
            refer_query = " ".join(refer_list)
            _, recommend = await self.db_manager.ai_db_search(refer_query, 
                                                              conversation)
            # 추천 질문 목록
            print("<다음과 같은 질문도 추천합니다.>")
            for dict_item in recommend:
                if len(recommend_list) >= 3:
                    break
                if dict_item["title"] not in refer_list:
                    recommend_list.append(dict_item["title"])
                    print(f"추천 질문: {dict_item['title']}")
            print("-"*50)

        refer_recom_list = (refer_list, recommend_list)
        return refer_recom_list, recommend

    async def _ai_recom(self, conversation, recommend):
        print("<AI가 추천하는 이런 질문은 어떠세요?>")
        if not recommend:
            query_text = "사람들이 자주 묻는 질문은 어떤 것들이 있어?"
            _, recommend = await self.db_manager.ai_db_search(query_text, 
                                                              conversation)
        context = "\n\n".join(
            [f"[{i+1}번] Question: {r['title']}:\nAnswer: {r['content']}" for i, r in enumerate(recommend)]
        )
        
        ai_recom_dict = await self.db_manager.ai_recommend(context, conversation)
        for title in ai_recom_dict["recommend"]:
            print(f"AI 추천 질문: {title}")
        print("-"*50)

        return ai_recom_dict
    
    def _build_messages(self, user_input, conversation, rag_results):
        
        system_prompt = """# Identity

당신은 스마트스토어 사장님 지원 상담 챗봇입니다.

# Instructions

* 상담사는 user의 질문에 대해 친절하고 정확하게 답변해야 합니다.
* 모든 메시지를 지속적인 대화의 일부로 간주하세요.
* user가 질문한 내용에 초점을 맞춰 답변을 유지하세요.
* 환각이나 꾸며낸 정보는 포함하지 마세요.

# Example Behavior

If the user says: "배송비가 얼마인가요?"  
→ You might respond: "기본 배송비는 3,000원입니다. 단, 제주 및 도서산간 지역은 추가 요금이 부과될 수 있습니다."  
""".strip()
        
        
        # RAG 프롬프트
        if rag_results:
            context = "\n\n".join(
                [
                    f"[{i+1}번] Question: {r['title']}:\nAnswer: {r['content']}"
                    for i, r in enumerate(rag_results)
                ]
            )
        else:
            context = ""

        rag_prompt = f"""# Instructions

다음은 다른 스마트스토어 고객들이 남긴 Q&A Records 입니다.  
user의 마지막 질문에 대해 Q&A Records의 내용을 참고하여 답변하세요.

# Context: Q&A Records

{context}

# Answer Guidelines

* user의 마지막 질문에 대해 Q&A Records의 내용만 참고하여 답변하세요.
* 항목이 여러 개인 경우 번호를 붙여 나열하세요.
* 필요 시 구분된 소제목(예: 배송 방법, 신청 서류)으로 나누어도 좋습니다.
* 답변은 반드시 "Q&A Records의 내용"에 근거해서 작성해야합니다.
* "Q&A Records의 내용"과 "user의 마지막 질문"이 서로 관련 없어서 답변이 불가능한 경우, 
  "{HIDDEN_TAG} 관련 내용이 없어 답변할 수 없습니다."라고만 답변하세요.
* Q&A Records에 없는 정보를 만들어내거나 가정하지 마세요.
* Q&A Records에서 사용하는 용어를 바꾸지 마세요.

# Output Format

* 답변에 참고한 Q&A Records가 있다면, 그 번호를 답변 마지막에 반드시 기입하세요. 
  Format: "{HIDDEN_TAG} [1, 3, 5]'
* 답변에 참고한 Q&A Records가 없다면, 빈 list를 반환하세요.
  Format: "{HIDDEN_TAG} []'

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
{HIDDEN_TAG} [4, 2]
</assistant_response>

<user_query>
네이버에서 사진 찍기 좋은 장소 알려줘줘 
</user_query>

<assistant_response>
{HIDDEN_TAG} 관련 내용이 없어 답변할 수 없습니다.
</assistant_response>
""".strip()

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation[-20:])
        messages.append({"role": "system", "content": rag_prompt})
        messages.append({"role": "user", "content": user_input})
        
        return messages