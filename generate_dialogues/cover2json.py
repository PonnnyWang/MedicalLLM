import json
import tiktoken

def convert_txt_to_json(file_path, save_path, model_name):
    '''
    txt文件转换为json文件并计算token
    '''
    encoding = tiktoken.encoding_for_model(model_name)
    entries = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            clean_line = line.strip()
            if clean_line:
                token_count = len(encoding.encode(clean_line))
                entry = {"tittle": "康复医学", "token": token_count, "desc": clean_line}
                entries.append(entry)

    with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(entries, json_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    file_path = r"/root/LLMproject/MedicalLLM/data/中国康复医学杂志_test.txt"
    save_path = r"/root/LLMproject/MedicalLLM/data/中国康复医学杂志_test.json"
    model_name = "gpt-3.5-turbo-0613"  

    convert_txt_to_json(file_path, save_path, model_name)

    # model_name in {  
    #         "gpt-3.5-turbo-0613",  
    #         "gpt-3.5-turbo-0301",
    #         "gpt-3.5-turbo-16k-0613",  
    #         "gpt-4-0314",  
    #         "gpt-4-32k-0314",  
    #         "gpt-4-0613",  
    #         "gpt-4-32k-0613",  
    #     }