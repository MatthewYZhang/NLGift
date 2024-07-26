import json
import random

keywords = ['network', 'graph', 'node', 'sort', 'length', 'order', 'linked', 'directed', 'weighted']

llama_format = '<s>[INST] <<SYS>>You are a helpful assistant.<</SYS>>{question}\nA:\n[/INST]{answer}</s>'

data = open('leetcode-train.jsonl', 'r').readlines()

target_file = open('graphcode.jsonl', 'w')

cnt = 0

for line in data:
    item = json.loads(line)
    # print(item)
    title_words = item['title'].lower()
    if (len(item['content']) + len(item['python']) < 4000) and any([keyword in title_words for keyword in keywords]):
        # print(item, end='\n\n')
        prompt = llama_format.format(question=item['content'], answer=item['python'])
        target_file.write(json.dumps({'text': prompt}) + '\n')
        cnt += 1
    if cnt == 200:
        break
