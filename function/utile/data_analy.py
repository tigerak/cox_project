import pickle
import re
from pprint import pprint
from collections import defaultdict
# modules
from config import *


### 데이터 전처리 도구 ###
def get_data(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    cleaned_data = {}

    for k, v in data.items():
        cleaned_key = common_clean(k)
        cleaned_value = filter_with_phrase(v)
        cleaned_data[cleaned_key] = cleaned_value

    return cleaned_data

def common_clean(text):
    cleaned_text = text.split("위 도움말이 도움이 되었나요?")[0]
    cleaned_text = cleaned_text.replace('\xa0', ' ').replace('\u200b', '').replace('\ufeff', '')
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def filter_with_phrase(text, phrase="위 도움말이 도움이 되었나요?"):
    if "위 도움말이 도움이 되었나요?" in text:
        cleaned_text = text.split(phrase)[0]
    else:
        print(f"도움이 되었나요 문구가 없습니다: {text}")
        cleaned_text = text
    return common_clean(cleaned_text)


### 데이터 분석 도구 ###
def data_analysis(keys):
    # print(f"Keys in the dataset: {keys}")
    result = parse_keys(keys)
    sets, counts = extract_set(result)
    # pprint(sets, width=100, indent=2)
    for v, c in counts.items():
        print(f"{v}: {len(c)}")
        for k, v in c.items():
            print(f"  {k}: {v}")

    print(f"Number of keys: {len(keys)}")
    # for key in keys:
    #     print(f"Key: {key}")
    #     print(f"Value: {data[key]}")

def parse_keys(key_list):
    parsed = []

    for key in key_list:
        item = {
            "category_1": None,
            "category_2": None,
            "title": None,
            "content_1": None,
            "content_2": None,
        }

        brackets = re.findall(r'\[(.*?)\]', key)
        if len(brackets) >= 1:
            item["category_1"] = brackets[0]
        if len(brackets) >= 2:
            item["category_2"] = brackets[1]

        # 각괄호 제거한 텍스트 추출
        key_wo_brackets = re.sub(r'\[.*?\]', '', key).strip()

        # title 및 괄호 추출
        parts = re.split(r'\((.*?)\)', key_wo_brackets)
        cleaned = re.sub(r'\s+', ' ', parts[0]).strip()
        cleaned = cleaned.replace('\u200b', '')
        item["title"] = cleaned

        if len(parts) > 3:
            item["content_1"] = parts[1].strip()
        if len(parts) >= 5:
            item["content_2"] = parts[3].strip()

        parsed.append(item)
    return parsed

def extract_set(parsed_list):
    # keys = ["category_1", "category_2", "title", "content_1", "content_2"]
    keys = ["category_1", "category_2", "content_1", "content_2"]
    result_set = {key: set() for key in keys}
    result_count = {key: defaultdict(int) for key in keys}

    for item in parsed_list:
        for key in keys:
            value = item.get(key)
            if value:
                result_set[key].add(value)
                result_count[key][value] += 1

    return result_set, result_count

