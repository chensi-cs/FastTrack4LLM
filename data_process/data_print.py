import json


def print_and_save_json(input_file, output_file):
    """Print the first 20 items of a JSON file and save the result."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        result = data[:20]  # First 20 elements if JSON is an array
    elif isinstance(data, dict):
        result = dict(list(data.items())[:20])  # First 20 key-value pairs if JSON is an object
    else:
        raise ValueError("Unsupported JSON format. Expected an array or object.")
    

    # print(json.dumps(result, indent=4, ensure_ascii=False))
    
    # Save the result
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    input_file = "../data/llm_data/processed/pretrain_hq.json" 
    output_file = "../data/llm_data/processed/demo_data/pretrain_hq_demo.json"  
    print_and_save_json(input_file, output_file)
