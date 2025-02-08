import json

def parse_json(response):
    try:
        left = response.index('{')
        right = response.rindex('}') + 1
        return json.loads(response[left:right])
    except (ValueError, json.JSONDecodeError) as e:
        print(f'Error parsing JSON: {e}')
        return {}

def gpus_needed(model_name: str) -> int:
    if "70b" in model_name.lower():
        return 4
    else:
        return 1