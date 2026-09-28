import argparse
import importlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser('Run one validated generated tool')
    parser.add_argument('--module', required=True)
    parser.add_argument('--input-json', required=True)
    parser.add_argument('--output-json', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    module_name = f'tools.generated.{args.module}'
    module = importlib.import_module(module_name)
    payload = json.loads(Path(args.input_json).read_text(encoding='utf-8'))
    result = module.run(
        context=payload.get('context', {}),
        **payload.get('args', {}),
    )
    if not isinstance(result, dict):
        result = {'result': result}
    Path(args.output_json).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
