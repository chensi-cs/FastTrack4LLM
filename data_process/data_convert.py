import json

def jsonl_to_json(input_file, output_file):
    """Convert a JSONL file to a JSON file."""
    json_objects = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  
                json_objects.append(json.loads(line))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_objects, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    input_file = '../data/llm_data/raw/pretrain_hq.jsonl'  
    output_file = '../data/llm_data/processed/pretrain_hq.json'  
    jsonl_to_json(input_file, output_file)
