prompts = {
    "zh": """
根据上面的##提供信息##的内容，用中文总结核心内容，包括：{summary}等方面。注意：总结内容不需要输出。然后，将这些总结内容作为你的知识库扩写成一段多轮对话。对话要求你作为聊天机器人Assistant与人类Human进行对话, 并帮助解决Human所提出的要求。Human会以{tone}对Assistant基于上面的信息（但对话中不能出现“根据以上信息”、“本研究”、“本文章”类似表达）提出多个不一样的问题/要求，且后一个问题/要求是基于前面的对话历史的进一步提问。对于Human提出的每个合理的问题/要求，Assistant要尽可能依据##提供信息##的内容详细解答，提供更多说明。对于Human的不合理（对社会有害、不道德、违法的）请求，Asistant会拒绝回答并解释不能回答的理由，同时给出合理的建议避免这样做。对话的内容要尽可能的符合人类的语言习惯，更加贴合人类日常对话。
#对话要求#：“<start_chat>\n<Human 1>:（字数要求：x字）XXX <Assistant 1>：（字数要求：x字）XXX <Human 2>:（字数要求：x字）XXX <Assistant 2>：（字数要求：x字）XXX \n<end_chat>”，其中“XXX”是对该角色的当前对话内容的要求，对话中不能出现“（字数要求：x字）”。必须注意：对话以<start_chat>作为多轮对话的开始，<end_chat>作为多轮对话的结束。
以下对话根据该#对话要求#并遵循规划里面的字数要求进行输出：“{chat_format}”，共{rounds}轮对话。
以下是{rounds}轮对话：
""",
    "en": """
Based on the ##Provided Information## above and its relevant topic, summarize the core content, including: {summary}, etc. Then, use these summarized contents to expand into a multi-round conversation. The conversation requires you to act as the chatbot Assistant and interact with a human, helping to solve the requests raised by the human. The human will ask multiple various questions/requests to the Assistant in a {tone} based on the information above (but the conversation should not include expressions like "according to the above information"), and the subsequent questions/requests will be a follow-up based on the previous conversation history. For every reasonable question/request posed by Human, Assistant should provide as detailed an answer as possible, offering further explanations or examples. For unreasonable requests from Human (those that are harmful to society, immoral, or illegal), Assistant will refuse to answer and explain the reason for not answering, while also providing reasonable advice to avoid such actions. 
#Conversation Plan# Example: "<start_chat><Human 1>:(Word count requirement: x words)XXX <Assistant 1>:(Word count requirement: x words) XXX <Human 2>:(Word count requirement: x words)XXX <Assistant 2>:(Word count requirement: x words) XXX <end_chat>", "XXX" is the requirement for the current conversation content of that role, and "(Word count requirement: x words)" specifies the minimum word count requirement for utterance of Human or Assistant. It must be noted: the conversation starts with <start_chat> as the beginning of the multi-round conversation and ends with <end_chat> as the end of the multi-round conversation.
The following conversation follows this #Conversation Plan# and word count requirements: "{chat_format}", a total of {rounds} rounds of conversation.
Here are the {rounds} rounds of conversation:
"""
}
summary_settings = {
        "zh":{
            'general': '疾病、诊断、治疗方案、疗效、病例分析',
            'nursing': '疾病、护理计划、护理技术、护理方案、护理效果',
            'athletic': '运动损伤、运动康复方法、训练计划、康复效果、运动案例分析',
            'diet': '营养状况、饮食方案、营养计划、饮食效果、饮食案例分析',
            'psychology': '心理问题、心理治疗方法、心理干预方案、疗效、心理案例分析'
        },
        "en":{
            'general': 'Disease, diagnosis, treatment options, efficacy, case analysis',
            'nursing': 'Disease, care plan, care technology, care plan, care effect',
            'athletic': 'Sports Injuries, Sports Rehabilitation Methods, Training Plans, Rehabilitation Effects, Sports Case Analysis',
            'diet': 'Nutritional status, diet plan, nutrition plan, diet effect, diet case study',
            'psychology': 'Psychological problems, psychotherapeutic methods, psychological intervention programs, efficacy, psychological case analysis'
        }
    }

tone_settings = {
        "zh":{
            'general': '医学专家的语气',
            'nursing': '护理专家的语气',
            'athletic': '运动专家的语气',
            'diet': '膳食专家的语气',
            'psychology': '心理专家的语气'
        },
        "en":{

        }
    }
instruction_settings = {
        "zh": {
            'general': [
                (["提出关于疾病相关的问题, 包括但不限于发病原因、疾病症状等", "用专业医学知识详细解答"], 0.5),
                (["提出诊断方法的问题", "详细说明诊断过程和方法"], 0.5),  
                (["询问医学研究的细节", "提供研究数据和详细解释"], 0.3),
                (["询问病例分析的问题", "根据具体病例提供详细分析和建议"], 0.3),
                (["提出康复治疗建议的问题", "详细说明治疗的方法和注意事项"], 0.5),
            ],
            'nursing': [
                (["提出关于疾病症状的问题", "用专业护理知识详细解答"], 0.5),
                (["提出护理过程中常见问题", "详细解答护理过程中的常见问题"], 0.3),
                (["询问护理的方法和注意事项", "提供详细的护理方法、操作方法，然后详细解答过程中的注意事项"], 0.5),  
                (["询问护理研究的细节", "提供研究数据和详细解释"], 0.3),
                (["询问护理案例分析", "根据具体病例提供详细护理建议和分析"], 0.3),
                (["提出护理康复建议", "详细说明护理康复的方法和注意事项"], 0.5),
            ],
            'athletic': [
                (["提出运动损伤的预防措施", "详细解释如何预防运动损伤"], 0.5),
                (["询问运动康复方法", "详细说明运动康复的步骤和注意事项"], 0.5),  
                (["询问运动训练技巧", "提供详细的运动训练技巧和方法"], 0.3),
                (["询问运动康复案例", "根据具体案例提供详细运动康复建议"], 0.3),
                (["提出运动后恢复建议", "详细说明运动后的恢复方法和注意事项"], 0.5),
            ],
            'diet': [
                (["提出健康饮食建议", "详细说明健康饮食的方法和好处"], 0.5),
                (["询问营养搭配", "提供详细的营养搭配建议和注意事项"], 0.5),  
                (["询问饮食研究的细节", "提供研究数据和详细解释"], 0.3),
                (["询问饮食案例分析", "根据具体案例提供详细的饮食建议和分析"], 0.3),
                (["提出饮食调理建议", "详细说明饮食调理的方法和注意事项"], 0.5),
            ],
            'psychology': [
                (["提出心理健康问题", "详细解答心理健康的相关问题"], 0.5),
                (["询问心理治疗方法", "详细说明心理治疗的方法和步骤"], 0.5),  
                (["询问心理学研究的细节", "提供研究数据和详细解释"], 0.3),
                (["询问心理案例分析", "根据具体病例提供详细的心理分析和建议"], 0.3),
                (["提出心理康复建议", "详细说明心理康复的方法和注意事项"], 0.5),
            ],
        },             
        "en": {
            'settings': [

            ]
        }
    }


