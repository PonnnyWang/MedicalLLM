#### 准备数据
* 准备txt文件，格式：
  ```
  text_line 1 
  text_line 2
  ...
  text_line n
  ```
* 使用`cover2json.py`将txt文件转换为jsonl文件。
#### 生成对话
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
    --language "zh" \             # 待处理的文件语言: ["zh", "en"]
    --instruction "general" \     # 生成数据集的风格: ["general", "nursing", "athletic", "diet", "psychology"]
    --assistant_word_count 300 \  
    --human_word_count 50 \       
    --num_turn_ratios 0.0 0.2 0.2 0.5 0.3  # 对话轮数概率
  ```

* `cd generate2datasets & bash run_script.sh`

#### 标准指令集构造
* ShareGPT格式（多轮对话）
  
  ```
  python dataset_builder.py --input_file "YOUR FILE" --output_path "YOUR SAVE PATHA" --data_format "sharegpt"
  ```
* Alpaca格式（指令跟随）
  ```
  python dataset_builder.py --input_file "YOUR FILE" --output_path "YOUR SAVE PATHA" --data_format "alpaca"
  ```
