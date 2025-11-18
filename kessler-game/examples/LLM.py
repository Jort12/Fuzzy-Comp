import textwrap
from openai import OpenAI
import re
from dotenv import load_dotenv
import os
import tempfile
import shutil

#load_dotenv()
#api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key='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') 

def gen_rule_set() -> str:
    prompt = """
        Create a ruleset that an ai agent has an aggressive play style.
        Return only valid Python code for the ruleset, do NOT include explanations or any libraries, just use the example as a reference on how to structure the code.
        Here is a list of Antecedents and Consequents:

        These are your Antecedents
        distance = 'very_close', 'close', 'sweet', 'far'
        rel_speed = 'slow', 'medium', 'fast'
        angle = 'left', 'ahead', 'fast'
        mine_distance = 'very_near', 'near', 'mid', 'far'
        mine_angle = 'left', 'ahead', 'fast'
        danger = 'imminent', 'risky', 'safe'

        These are your Consequents
        thrust = 'reverse_hard', 'reverse_soft', 'medium', 'high'
        turn = 'hard_left', 'soft_left', 'zero', 'soft_right', 'hard_right'
        fire = 'no', 'yes'
        mine = 'no', 'yes'

        Use this code as a reference to creating the rule set:
        rules = []
        rules += [
            ctrl.Rule(mine_distance['very_near'] & mine_angle['left'],  (thrust['high'],   turn['hard_right'], fire['no'], mine['no'])),
            ctrl.Rule(mine_distance['very_near'] & mine_angle['right'], (thrust['high'],   turn['hard_left'],  fire['no'], mine['no'])),
            ctrl.Rule(mine_distance['very_near'] & mine_angle['ahead'], (thrust['high'],   turn['soft_right'], fire['no'], mine['no'])),
        ]

        For refernce the current ruleset contains 32 different rules, it doesn't necessarily need to be 32 but the range of rules should 
        be 20 - 45 rules.

        GENERATE PYTHON CODE ONLY and Thank you.
    """
    
    response = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )

    full_text = response.output[0].content[0].text
    match = re.search(r"```python\s*(.*?)```", full_text, re.DOTALL)

    if match:
        code_only = match.group(1).strip()
    else:
        code_only = ""

    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(code_only + "\n")

    return code_only

def insert_gen_code(code: str):
    file_path = (
        r"C:\Users\andje\Documents\GitHub\Fuzzy-Comp"
        r"\kessler-game\examples\fuzzy_aggressive_controller.py"
    )
    starter_marker = "#BEGIN GENERATED CODE"
    end_marker = "#END GENERATED CODE"

    # Normalize generated code indentation
    clean_code = textwrap.dedent(code).strip("\n")

    # Read entire file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find markers
    start = content.find(starter_marker)
    end = content.find(end_marker)

    if start == -1 or end == -1:
        raise ValueError("Generated code markers not found")

    # Locate indent based on following line
    start_line_end = content.find("\n", start) + 1
    next_line_end = content.find("\n", start_line_end)
    if next_line_end == -1:
        next_line_end = len(content)

    next_line = content[start_line_end:next_line_end]
    indent_str = next_line[:len(next_line) - len(next_line.lstrip())]

    # Apply indentation to new code
    indented_code = "\n".join(
        indent_str + line if line.strip() else line
        for line in clean_code.splitlines()
    )

    # Apply indentation to end marker too
    indented_end_marker = indent_str + end_marker

    # Rebuild file content
    updated = (
        content[:start_line_end]
        + indented_code + "\n"
        + indented_end_marker
        + content[end + len(end_marker):]
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(updated)3