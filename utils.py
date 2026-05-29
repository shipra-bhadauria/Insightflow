import json


def safe_serialize(obj):
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_serialize(i) for i in obj]
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    else:
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)


def safe_json(obj) -> str:
    return json.dumps(safe_serialize(obj))