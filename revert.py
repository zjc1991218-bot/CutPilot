import json

with open('/Users/zhangjiacheng/.gemini/antigravity/brain/9314fa29-109e-4daa-ad9f-2e1fa93b5e39/.system_generated/logs/transcript_full.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('step_index') == 38:
            tool_calls = data.get('tool_calls', [])
            for tc in tool_calls:
                if tc['name'] == 'write_to_file':
                    code_content = tc['args']['CodeContent']
                    with open('/Users/zhangjiacheng/Desktop/video_cutter/app.py', 'w') as out_f:
                        out_f.write(code_content)
                    print("Reverted app.py successfully!")
                    exit(0)
