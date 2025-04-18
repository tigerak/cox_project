import re
import json
# ChromaDB
import chromadb


class ChromaDB:
    def __init__(self):
        client = chromadb.PersistentClient(path="./smart/data/chroma_db")

        # self.collection = client.get_or_create_collection(name="test_manual")
        self.collection = client.get_or_create_collection(name="consult_manual")


    def add_data(self, id, title, content, embedding):
        self.collection.upsert(
            documents=[title],
            embeddings=[embedding],
            metadatas=[{"content": content}],
            ids=[id]
        )
    
### json 파싱 유틸 ###
def safe_parse_json(text):
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
            # "include": [...] 와 "exclude": [...] 추출
            inc_match = re.search(r'"include"\s*:\s*\[[^\]]*\]', json_str)
            exc_match = re.search(r'"exclude"\s*:\s*\[[^\]]*\]', json_str)
            if inc_match and exc_match:
                recovered = '{' + inc_match.group() + ', ' + exc_match.group() + '}'
                try:
                    return json.loads(recovered)
                except json.JSONDecodeError as e3:
                    print("복구된 문자열 파싱 실패:", e3)
                
        raise ValueError("유효한 JSON 형식을 찾을 수 없습니다.")
        
def safe_parse_json_2(text):
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
            # "include": [...] 와 "exclude": [...] 추출
            recom_match = re.search(r'"recommend"\s*:\s*\[[^\]]*\]', json_str)
            if recom_match:
                recovered = '{' + recom_match.group() + '}'
                try:
                    return json.loads(recovered)
                except json.JSONDecodeError as e3:
                    print("복구된 문자열 파싱 실패:", e3)
                
        raise ValueError("유효한 JSON 형식을 찾을 수 없습니다.")
    
def fix_extra_closing_brace(text: str) -> str:
    """
    문장 끝 중복 중괄호 제거
    """
    open_braces = text.count('{')
    close_braces = text.count('}')
    # 닫는 중괄호가 더 많다면 초과만큼 제거
    if close_braces > open_braces:
        diff = close_braces - open_braces
        for _ in range(diff):
            # 뒤에서부터 하나씩 제거
            text = text[::-1].replace('}', '', 1)[::-1]
    return text