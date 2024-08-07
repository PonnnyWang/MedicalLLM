import os
import openai
import json
import logging
import random
import time
import numpy as np
from typing import Dict, Tuple, List
from dotenv import load_dotenv, find_dotenv
import argparse
from tqdm import tqdm
from openai import OpenAI
from configs import *
from concurrent.futures import ThreadPoolExecutor, as_completed

_ = load_dotenv(find_dotenv()) 
openai.api_key  = os.getenv('OPENAI_API_KEY')

random.seed()
np.random.seed()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DialogueGenerator:
    def __init__(self, model: str = "gpt-4o-mini"): # gpt-3.5-turbo-0613/gpt-4/gpt-4-turbo-preview
        self.model = model
        self.task_id_generator = task_id_generator()
        self.context = {}  
    
    def get_completions(self, system_input: str, user_input: str) -> str:
        """使用指定模型生成对话"""
        max_retries = 5
        retry_delay = 60  # in seconds
        attempts = 0
        while attempts < max_retries:
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    temperature=0.7,
                    messages=[
                        {
                            "role": "system", 
                            "content": system_input
                        }, 
                        {   
                            "role": "user", 
                            "content": user_input
                        }]
                )
                return response
            except Exception as e:
                attempts += 1
                logging.error(f"Error generating dialogue (attempt {attempts}/{max_retries}): {e}")
                if attempts < max_retries:
                    time.sleep(retry_delay)
        return None

    def encode_prompt(self, context: Dict, rounds: Tuple[int] = None, word_counts: Dict = None, language: str = "zh", instruction: str = "general") -> Tuple[str, str]:
        """编码提示"""
        if language == "zh":
            system_input =  "要求你作为聊天机器人Assistant与人类Human进行多轮对话。对话是根据##提供信息##的内容开展的，并以#对话要求#的格式进行输出，以<start_chat>开始，以<end_chat>结束。"
        else:
            system_input = "You are asked to chat with a human as a chatbot Assistant in multiple rounds. The dialogue is based on the ##Provided Information## and is output in the format of #Conversation Plan#, starting with <start_chat> and ending with <end_chat>."

        if rounds is None and word_counts is None:
            selected_round = [2, 3, 4, 5]
            rounds = random.choices(selected_round, weights=[0.0, 0.5, 0.3, 0.2])[0]
            word_counts = [100] * rounds
        if rounds is None and word_counts is not None:
            rounds = len(word_counts)
            
        user_input = ""
        chat_format = ""
        chat_format += "<start_chat>\n"

        local_settings = list(zip(*instruction_settings[language][instruction]))
        human_word_counts = word_counts['human']
        assistant_word_counts = word_counts['assistant']

        for i in range(rounds):
            if human_word_counts[i] < 10: human_word_counts[i] = 30
            if assistant_word_counts[i] < 100: assistant_word_counts[i] = 200
            requirements = random.choices(local_settings[0], weights=local_settings[1], k=1)[0]
            if i == 0:
                chat_format += f"<Human {i+1}>：（字数要求：{human_word_counts[i]}字）{requirements[0]} \n<Assistant {i+1}>：" if language == "zh" else f"<Human {i+1}>:(word count: {human_word_counts[i]} words){requirements[0]} <Assistant {i+1}>:"
            else:
                chat_format += f"<Human {i+1}>：（字数要求：{human_word_counts[i]}字）进一步{requirements[0]} \n<Assistant {i+1}>：" if language == "zh" else f"<Human {i+1}>:(word count: {human_word_counts[i]} words)further {requirements[0]} <Assistant {i+1}>:"
            chat_format  += f"（字数要求：{assistant_word_counts[i]}字）{requirements[1]} " if language == "zh" else f"(word count: {assistant_word_counts[i]} words){requirements[1]} "

        chat_format  += "\n<end_chat>"
        summary = summary_settings[language][instruction]
        tone = tone_settings[language][instruction]
        prompt = prompts[language].format(summary=summary, tone=tone, chat_format=chat_format, rounds=rounds)

        user_input += f"##提供信息##\n" if language == "zh" else f"##Provided Information##\n"
        user_input += context['desc']
        user_input += f"\n\n"
        user_input += prompt
        user_input += f"\n\n"
        user_input += f"##输出检查##\n 在输出对话之前检查格式是否符合要求，如果不符合，请调整为正确格式后输出。" if language == "zh" else f"##Output Check##\n Ensure the output format is correct before providing the conversation. Adjust if necessary."
        return system_input, user_input, prompt, rounds

    def post_process_gpt_response(self, response: Dict) -> str:
        """后处理GPT响应"""
        response = response.choices[0]
        try:
            raw_chat = response.message.content
        except:
            print("ERROR parse!")
            return None  
        if not raw_chat.startswith('<start_chat>') or not raw_chat.endswith('<end_chat>'):
            return None
        return raw_chat

    def _generate_dialogue(self, context: Dict, args) -> Dict:
        """生成单个对话"""
        selected_round = [1, 2, 3, 4, 5]
        rounds = random.choices(selected_round, weights=args.num_turn_ratios)[0]  # number of turns in the dialogue       
        assistant_word_counts = (np.random.normal(loc=args.assistant_word_count, scale=50, size=rounds).astype(int) // 50 * 50).tolist()
        human_word_counts = (np.random.normal(loc=args.human_word_count, scale=50, size=rounds).astype(int) // 50 * 50).tolist()
        word_counts = {
            "assistant": assistant_word_counts,
            "human": human_word_counts
        }
        system_input, user_input, prompt, rounds = self.encode_prompt(context, 
                                                                    rounds=rounds, 
                                                                    word_counts=word_counts,
                                                                    language=args.language,
                                                                    instruction=args.instruction)
        response = self.get_completions(system_input, user_input)
        if response:
            chat = self.post_process_gpt_response(response)
            token_count = response.usage.total_tokens
            if chat:
                task_id = next(self.task_id_generator)
                return {
                    "id": task_id, 
                    "total_tokens": token_count,
                    "prompt": prompt,    
                    "dialogue": chat
                }
        return None

    def generate_dialogues(self, contexts: List[Dict], args):
        """批量生成对话"""
        dialogues = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            futures = [executor.submit(self._generate_dialogue, context, args) for context in contexts]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
                result = future.result()
                try:
                    result = future.result()
                    if result:
                        dialogues.append(result)
                except Exception as e:
                    print(f"An error occurred: {e}")
            print("All generate tasks have been completed!")
        dialogues.sort(key=lambda x: x['id'])
        return dialogues

def task_id_generator():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1

def save_dialogues_to_json(dialogues, output_file):
    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=2)
   
def save_dialogues_to_jsonl(dialogues, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for dialogue in dialogues:
            json.dump(dialogue, f, ensure_ascii=False, indent=2)
            f.write("\n")

def main(args):
    contexts = []
    with open(args.file_path, "r", encoding="utf-8") as f:
        for line in f:
            context = json.loads(line)
            contexts.append(context)
    generator = DialogueGenerator()
    dialogues = generator.generate_dialogues(contexts, args)
    save_dialogues_to_json(dialogues, args.save_path)
    print("All tasks are saved!")
    # save_dialogues_to_jsonl(dialogues, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, \
                        help="json files containing references")
    parser.add_argument("--save_path", type=str, \
                        help="json file to save results to")
    parser.add_argument("--language", default="zh", \
                        help='Language of the generated dialogue. "zh" for Chinese, "en" for English.', choices=["zh", "en"])
    parser.add_argument("--instruction", type=str, default="general", \
                        help='What domain knowledge is used to generate dialogue.', choices=["general", "nursing", "athletic", "diet", "psychology"])
    parser.add_argument("--assistant_word_count", type=int, default=200, \
                        help='Number of words for the assistant to generate')
    parser.add_argument("--human_word_count", type=int, default=30, \
                        help='Number of words for the human to generate')
    parser.add_argument("--num_turn_ratios", nargs="+", type=float, default=[0, 0, 0.5, 0.5, 0], \
                        help='Ratio of the number of turns in the dialogue. The first number is the ratio of 1-turn dialogue, the second number is the ratio of 2-turn dialogue, and so on.')
    args = parser.parse_args()

    main(args)
