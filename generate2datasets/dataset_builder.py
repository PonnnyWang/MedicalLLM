import os
import json
import argparse
import logging
import re

logging.basicConfig(filename='error_log.log', level=logging.ERROR, format='%(asctime)s - %(message)s')

def convert_to_sharegpt(data):
    sharegpt_data = []
    for item in data:
        task_id = item["id"]
        dialogue = re.split(r'(?<!\n)\n(?!\n)|(?<=\n\n)(?=<)', item["dialogue"])
        conversation = []
        try:
            for line in dialogue:
                if line.startswith("<Human") or line.startswith("<Assistant"):
                    role, text = re.split("：|:", line, 1)
                    text = re.sub(r'^（字数要求：\d+字）', '', text).strip()
                    role = "human" if "Human" in role else "assistant"
                    conversation.append({"from": role, "value": text.strip()})
            sharegpt_data.append({"conversations": conversation})
        except Exception as e:
            logging.error(f"task {task_id} conversion error: {e}")
            continue
    return sharegpt_data

def convert_to_alpaca(data):
    alpaca_data = []
    for item in data:
        task_id = item["id"]
        dialogue = item["dialogue"].split("\n")
        history = []
        instruction = "请以专业的医学知识进行回答。"
        system = "要求你作为聊天机器人Assistant与人类Human进行多轮对话。"
        last_human = ""
        last_assistant = ""
        try:
            for line in dialogue:
                if line.startswith("<Human"):
                    role, text = re.split("：|:", line, 1)
                    text = re.sub(r'^（字数要求：\d+字）', '', text).strip()
                    if last_human:  
                        history.append([last_human, last_assistant])
                    last_human = text
                elif line.startswith("<Assistant"):
                    role, text = re.split("：|:", line, 1)
                    text = re.sub(r'^（字数要求：\d+字）', '', text).strip()
                    last_assistant = text
            # 最后一轮对话作为input和output
            input_text = last_human
            output_text = last_assistant
            alpaca_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "system": system,
                "history": history 
            })
        except Exception as e:
            logging.error(f"task {task_id} conversion error: {e}")
            continue
    return alpaca_data

def save_file(args):
    with open(args.input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    if args.data_format == "sharegpt":
        instruction_data = convert_to_sharegpt(data)
    else:
        instruction_data = convert_to_alpaca(data)
    output_file = os.path.join(args.output_path, f"{args.data_format}_dataset.json")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(instruction_data, outfile, indent=2, ensure_ascii=False)
    print(f"The {args.data_format} instruction dataset is successfully constructed !")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, \
                        help="JSON file that needs to be converted")
    parser.add_argument("--output_path", type=str, \
                        help="instruction datasets file is saved to")
    parser.add_argument("--data_format", type=str, default="sharegpt", \
                        help="The format of the dataset, choose sharegpt or alpaca")

    args = parser.parse_args()
    save_file(args)
