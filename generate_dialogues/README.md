#### 运行
* 设置OpenAI API环境变量：
    
    `pip install python-dotenv & openai`
    ```
    export OPENAI_API_KEY="your_api_key"
    export OPENAI_BASE_URL="base_url"
    ```
* 修改`run_script.sh`中的输入参数。
  ```
    --file_path "XXX" \  
    --save_path "XXX" \
    --language "zh" \             # 文件语言
    --assistant_word_count 300 \  # 单轮助手最多对话字数
    --human_word_count 50 \       # 单轮用户最多对话字数
    --num_turn_ratios 0 0 0.2 0.5 0.3  # 对话轮数概率
  ```

* `cd generate_dialogues.py & bash run_script.sh`
