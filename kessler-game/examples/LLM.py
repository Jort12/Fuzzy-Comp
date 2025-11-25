import textwrap
from openai import OpenAI
import re
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) 

def gen_rule_set() -> str:
    prompt = """
    You are an expert in fuzzy control systems and Python code generation.
    Your task is to generate NEW fuzzy rule combinations based on the rule set below.

    GOAL:
    - Produce fewer or alternate combinations.
    - Recombine conditions (mine_distance, mine_angle, angle, danger, distance, rel_speed).
    - Maintain logical correctness for a defensive control system.
    - Keep the same general style: ctrl.Rule(condition, output_tuple)
    - All outputs must be valid Python code using ctrl.Rule().
    - You MAY reduce, simplify, or merge rules.
    - You MAY restructure logic (e.g., combine OR patterns).
    - The new rules must still be realistic and consistent.

    STRICT OUTPUT RULES:
    - Return ONLY Python code.
    - No comments.
    - No markdown.
    - No explanations.
    - Only a `rules = []` block with appended ctrl.Rule() entries.

    EXISTING RULESET:
    rules = []

    rules += [
        ctrl.Rule(mine_distance['very_near'] & mine_angle['left'],  (thrust['high'],   turn['hard_right'], fire['no'], mine['no'])),
        ctrl.Rule(mine_distance['very_near'] & mine_angle['right'], (thrust['high'],   turn['hard_left'],  fire['no'], mine['no'])),
        ctrl.Rule(mine_distance['very_near'] & mine_angle['ahead'], (thrust['high'],   turn['soft_right'], fire['no'], mine['no'])),
    ]

    rules += [
        ctrl.Rule(mine_distance['near'] & mine_angle['left'],  (thrust['high'],   turn['hard_right'], mine['no'])),
        ctrl.Rule(mine_distance['near'] & mine_angle['right'], (thrust['high'],   turn['hard_left'],  mine['no'])),
        ctrl.Rule(mine_distance['near'] & mine_angle['ahead'], (thrust['high'],   turn['soft_right'], mine['no'])),
        ctrl.Rule(mine_distance['near'] & angle['ahead'], fire['yes']),
    ]

    rules += [
        ctrl.Rule(mine_distance['mid'] & mine_angle['left'],  (thrust['medium'], turn['soft_right'])),
        ctrl.Rule(mine_distance['mid'] & mine_angle['right'], (thrust['medium'], turn['soft_left'])),
        ctrl.Rule(mine_distance['mid'] & angle['ahead'], fire['yes']),
    ]

    rules += [
        ctrl.Rule(danger['imminent'] & angle['left'],  (thrust['reverse_hard'], turn['hard_right'], fire['no'], mine['no'])),
        ctrl.Rule(danger['imminent'] & angle['right'], (thrust['reverse_hard'], turn['hard_left'],  fire['no'], mine['no'])),
        ctrl.Rule(danger['imminent'] & angle['ahead'], (thrust['reverse_hard'], turn['soft_right'], fire['no'], mine['no'])),
        ctrl.Rule(danger['risky'], thrust['medium']),
    ]

    rules += [
        ctrl.Rule(distance['very_close'], thrust['reverse_hard']),
        ctrl.Rule(distance['very_close'] & angle['ahead'], turn['soft_right']),
        ctrl.Rule(distance['close'] & rel_speed['fast'], thrust['reverse_soft']),
    ]

    rules += [
        ctrl.Rule((angle['ahead']) & (danger['safe'] | danger['risky']) & (mine_distance['far'] | mine_distance['mid'] | mine_distance['near']), fire['yes'])
    ]

    rules.append(ctrl.Rule(mine_distance['far'] & distance['very_close'], (thrust['reverse_hard'], turn['zero'], fire['no'])))

    rules += [
        ctrl.Rule(mine_distance['far'] & (danger['safe'] | danger['risky']) & distance['close'] & angle['ahead'], (thrust['medium'], turn['zero'], fire['yes'])),
        ctrl.Rule(mine_distance['far'] & (danger['safe'] | danger['risky']) & distance['close'] & angle['left'],  (thrust['medium'], turn['soft_left'])),
        ctrl.Rule(mine_distance['far'] & (danger['safe'] | danger['risky']) & distance['close'] & angle['right'], (thrust['medium'], turn['soft_right'])),
    ]

    rules += [
        ctrl.Rule(mine_distance['far'] & danger['safe'] & distance['sweet'] & angle['ahead'], (thrust['medium'], turn['zero'], fire['yes'])),
        ctrl.Rule(mine_distance['far'] & danger['safe'] & distance['sweet'] & (angle['left'] | angle['right']), (thrust['medium'], fire['yes'])),
    ]

    rules += [
        ctrl.Rule(mine_distance['far'] & (danger['safe'] | danger['risky']) & distance['far'] & angle['ahead'], (thrust['high'], turn['zero'], fire['yes'])),
        ctrl.Rule(mine_distance['far'] & (danger['safe'] | danger['risky']) & distance['far'] & angle['left'],  (thrust['high'], turn['soft_left'])),
        ctrl.Rule(mine_distance['far'] & (danger['safe'] | danger['risky']) & distance['far'] & angle['right'], (thrust['high'], turn['soft_right'])),
    ]

    rules.append(ctrl.Rule(mine_distance['far'] & (distance['sweet'] | distance['far']) & rel_speed['fast'] & angle['ahead'], fire['yes']))

    rules += [
        ctrl.Rule(mine_distance['far'] & danger['safe'] & (distance['close'] | distance['sweet']) & angle['ahead'], mine['yes']),
        ctrl.Rule(mine_distance['far'] & (distance['close'] | distance['sweet']) & rel_speed['fast'], mine['yes']),
        ctrl.Rule(mine_distance['very_near'] | mine_distance['near'] | danger['imminent'], mine['no']),
    ]
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
        r"C:\Users\Orteg\OneDrive\Pictures\Documents\GitHub\Fuzzy-Comp"
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
        f.write(updated)