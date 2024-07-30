import os
import openai
import json
import logging
import random
import numpy as np
from typing import Dict, Tuple, List
from dotenv import load_dotenv, find_dotenv
import argparse
from tqdm import tqdm
from openai import OpenAI

_ = load_dotenv(find_dotenv()) 
openai.api_key  = os.getenv('OPENAI_API_KEY')

random.seed(13)
np.random.seed(13)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DialogueGenerator:
    def __init__(self, model: str = "gpt-4"): # gpt-3.5-turbo-0613/gpt-4
        self.model = model
        self.task_id_generator = task_id_generator()
        self.context = {}  
    
    def generate_dialogue(self, system_input: str, user_input: str) -> str:
        """使用指定模型生成对话"""
        try:
            response = openai.chat.completions.create(
                model=self.model,
                temperature=0.7,
                messages=[
                    {
                        "role": "system", 
                        "content": system_input
                    }, 
                    {   "role": "user", 
                        "content": user_input
                    }]
            )
            return response
        except Exception as e:
            logging.error(f"Error generating dialogue: {e}")
            return None

    def encode_prompt(self, context: Dict, rounds: Tuple[int] = None, word_counts: Dict = None, language: str = "zh") -> Tuple[str, str]:
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
        chat_format += "<start_chat>"

        local_settings = {
            "zh": {
                'settings': [
                    (["以医学专家的语气提问", "以医学专业术语和详细解释回答"], 0.3),
                    (["提出关于疾病症状的问题", "用专业医学知识详细解答"], 0.5),
                    (["提出诊断方法的问题", "详细说明诊断过程和方法"], 0.5),  
                    (["询问医学研究的细节", "提供研究数据和详细解释"], 0.3),
                    (["询问病例分析的问题", "根据具体病例提供详细分析和建议"], 0.3),
                    (["提出康复治疗建议的问题", "详细说明治疗的方法和注意事项"], 0.5),
                ],
            },             
            "en": {
                'settings': [
                    (["asks in a medical expert's tone", "answers with medical terminology and detailed explanation"], 0.5),
                    (["asks about symptoms of diseases", "answers with professional medical knowledge"], 0.5),
                    (["requests medical advice", "provides detailed medical suggestions and explanations"], 0.5),
                    (["inquires about details of medical research", "provides research data and detailed explanation"], 0.5),
                    (["asks about medication suggestions", "explains the effects and precautions of the medication in detail"], 0.5),
                ]
                    
            }
        }

        local_settings = list(zip(*local_settings[language]['settings']))
        human_word_counts = word_counts['human']
        assistant_word_counts = word_counts['assistant']

        for i in range(rounds):
            if human_word_counts[i] < 10: human_word_counts[i] = 20
            if assistant_word_counts[i] < 100: assistant_word_counts[i] = 200
            requirements = random.choices(local_settings[0], weights=local_settings[1], k=1)[0]
            if i == 0:
                chat_format += f"<Human {i+1}>：（字数要求：{human_word_counts[i]}字）{requirements[0]} <Assistant {i+1}>：" if language == "zh" else f"<Human {i+1}>:(word count: {human_word_counts[i]} words){requirements[0]} <Assistant {i+1}>:"
            else:
                chat_format += f"<Human {i+1}>：（字数要求：{human_word_counts[i]}字）进一步{requirements[0]} <Assistant {i+1}>：" if language == "zh" else f"<Human {i+1}>:(word count: {human_word_counts[i]} words)further {requirements[0]} <Assistant {i+1}>:"
            chat_format  += f"（字数要求：{assistant_word_counts[i]}字）{requirements[1]} " if language == "zh" else f"(word count: {assistant_word_counts[i]} words){requirements[1]} "

        chat_format  += "<end_chat>"
        
        if language == "zh":
            prompt = \
    f"""
    根据上面的##提供信息##的内容，用中文总结核心内容，包括：疾病、诊断、治疗方案、疗效、病例分析等方面。注意：总结内容不需要输出。然后，将这些总结内容作为你的知识库扩写成一段多轮对话。对话要求你作为聊天机器人Assistant与人类Human进行对话, 并帮助解决Human所提出的要求。Human会以人类的语气对Assistant基于上面的信息（但对话中不能出现“根据以上信息”、“本研究”、“本文章”类似表达）提出多个不一样的问题/要求，且后一个问题/要求是基于前面的对话历史的进一步提问。对于Human提出的每个合理的问题/要求，Assistant要尽可能依据##提供信息##的内容详细解答，提供更多说明。对于Human的不合理（对社会有害、不道德、违法的）请求，Asistant会拒绝回答并解释不能回答的理由，同时给出合理的建议避免这样做。对话的内容要尽可能的符合人类的语言习惯，更加贴合人类日常对话。
    #对话要求#：“<start_chat><Human 1>:（字数要求：x字）XXX <Assistant 1>：（字数要求：x字）XXX <Human 2>：（字数要求：x字）XXX <Assistant 2>：（字数要求：x字）XXX <end_chat>”，其中“XXX”是对该角色的当前对话内容的要求，“（字数要求：x字）”是Human或者Assistant说话的最低字数要求, （字数要求：x字）XXX 是给你的提示，对话内容中不应该出现。必须注意：对话以<start_chat>作为多轮对话的开始，<end_chat>作为多轮对话的结束。
    以下对话根据该#对话要求#并遵循规划里面的字数要求进行输出：“{chat_format}”，共{rounds}轮对话。
    示例对话：
    <start_chat>
    <Human 1>：脊髓损伤会给人体带来哪些不利影响？
    <Assistant 1>：脊髓损伤会给人体带来多种不利影响。首先是运动功能障碍，患者可能会出现肢体瘫痪或无力。其次是感觉障碍，患者可能会感到疼痛、麻木或对温度的感觉减弱。此外，脊髓损伤还可能导致自主神经功能障碍，影响排尿、排便以及性功能。心理方面，患者常常会因为身体上的变化产生焦虑和抑郁情绪。
    <Human 2>：如何有效治疗脊髓损伤？
    <Assistant 2>：<Assistant 2>：治疗脊髓损伤的方法多种多样，主要包括手术治疗、药物治疗和康复治疗。手术治疗一般用于解除脊髓压迫，修复脊髓周围的组织。药物治疗主要包括抗炎药、止痛药和神经营养药等，用于减轻症状和促进恢复。康复治疗则包括物理治疗、作业治疗和心理治疗，帮助患者恢复功能，提高生活质量。此外，早期干预和持续的康复训练对改善预后至关重要。
    ...
    <end_chat>
    """
            prompt += f"以下是{rounds}轮对话："
        else:
            prompt = \
    f"""
    Based on the ##Provided Information## above and its relevant topic, summarize the core content, including: diseases, treatment plans, effects, case analysis, etc. Then, use these summarized contents to expand into a multi-round conversation. The conversation requires you to act as the chatbot Assistant and interact with a human, helping to solve the requests raised by the human. The human will ask multiple various questions/requests to the Assistant based on the information above (but the conversation should not include expressions like "according to the above information"), and the subsequent questions/requests will be a follow-up based on the previous conversation history. For every reasonable question/request posed by Human, Assistant should provide as detailed an answer as possible, offering further explanations or examples. For unreasonable requests from Human (those that are harmful to society, immoral, or illegal), Assistant will refuse to answer and explain the reason for not answering, while also providing reasonable advice to avoid such actions. 
    #Conversation Plan# Example: "<start_chat><Human 1>:(Word count requirement: x words)XXX <Assistant 1>:(Word count requirement: x words) XXX <Human 2>:(Word count requirement: x words)XXX <Assistant 2>:(Word count requirement: x words) XXX <end_chat>", "XXX" is the requirement for the current conversation content of that role, and "(Word count requirement: x words)" specifies the minimum word count requirement for utterance of Human or Assistant. It must be noted: the conversation starts with <start_chat> as the beginning of the multi-round conversation and ends with <end_chat> as the end of the multi-round conversation.
    The following conversation follows this #Conversation Plan# and word count requirements: "{chat_format}", a total of {rounds} rounds of conversation.
    """
            prompt += f"Here are the {rounds} rounds of conversation:"
                
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
        # if not raw_chat.startswith('<start_chat>') or not raw_chat.endswith('<end_chat>'):
        #     return None
        return raw_chat

    def generate_dialogues(self, contexts: List[Dict], args):
        """批量生成对话"""
        dialogues = []
        selected_round = [1, 2, 3, 4, 5]
        rounds = random.choices(selected_round, weights=args.num_turn_ratios)[0]  # number of turns in the dialogue       
        assistant_word_counts = (np.random.normal(loc=args.assistant_word_count, scale=50, size=rounds).astype(int) // 50 * 50).tolist()
        human_word_counts = (np.random.normal(loc=args.human_word_count, scale=50, size=rounds).astype(int) // 50 * 50).tolist()
        word_counts = {
            "assistant": assistant_word_counts,
            "human": human_word_counts
        }
        with tqdm(total=len(contexts), desc="generate dialogues", unit="dialogues") as pbar:
            for context in contexts:
                system_input, user_input, prompt, rounds = self.encode_prompt(context, 
                                                                            rounds=rounds, 
                                                                            word_counts=word_counts,
                                                                            language=args.language
                                                                            )
                response = self.generate_dialogue(system_input, user_input)
                if response:
                    chat = self.post_process_gpt_response(response)
                    token_count = response.usage.total_tokens
                    if chat:
                        task_id = next(self.task_id_generator)
                        dialogues.append({
                            "id": task_id, 
                            "total_tokens": token_count,
                            "prompt": prompt,    
                            "dialogue": chat
                        })
                        print(f"dialogue {task_id} is generated successfully!")
                pbar.update(1)        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, \
                        help="json files containing references")
    parser.add_argument("--save_path", type=str, \
                        help="json file to save results to")
    parser.add_argument("--language", default="zh", \
                        help='Language of the generated dialogue. "zh" for Chinese, "en" for English.', choices=["zh", "en"])
    parser.add_argument("--assistant_word_count", type=int, default=200, \
                        help='Number of words for the assistant to generate')
    parser.add_argument("--human_word_count", type=int, default=30, \
                        help='Number of words for the human to generate')
    parser.add_argument("--num_turn_ratios", nargs="+", type=float, default=[0, 0, 0.5, 0.5, 0], \
                        help='Ratio of the number of turns in the dialogue. The first number is the ratio of 1-turn dialogue, the second number is the ratio of 2-turn dialogue, and so on.')
    args = parser.parse_args()

    with open(args.file_path, "r", encoding="utf-8") as f:
        contexts = json.load(f)
    generator = DialogueGenerator()
    dialogues = generator.generate_dialogues(contexts, args)
    save_dialogues_to_json(dialogues, args.save_path)
    # save_dialogues_to_jsonl(dialogues, args.save_path)
