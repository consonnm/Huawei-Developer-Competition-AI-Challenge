import json

# JSONL 文件的路径
jsonl_file_path = 'train.jsonl'
# 将提取的输入保存到 TXT 文件的路径
txt_file_path = 'knowledge_input.txt'

# 读取 JSONL 文件并提取 'input' 字段
with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file, \
     open(txt_file_path, 'w', encoding='utf-8') as txt_file:
    for line in jsonl_file:
        # 解析 JSONL 文件中的每一行
        record = json.loads(line)
        # 获取 'input' 字段并写入到 TXT 文件
        input_text = record.get('input', '')  # 使用 .get() 以防某些行没有 'input' 字段
        output_txt=record.get('output', '')
        txt_file.write(input_text + '\n')  # 添加换行符，以便每个输入都在 TXT 文件中的新一行
       

print(f"Inputs have been extracted to {txt_file_path}")
