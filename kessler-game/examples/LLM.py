from openai import OpenAI
import re
from dotenv import load_dotenv
import os
import tempfile
import shutil

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) 

def gen_rule_set() -> str:

    print("im here in gen_rule_set\n")

    prompt = """
        Create a ruleset that the agent has an aggressive play style.
        Return only valid Python code for the ruleset, do NOT include explanations.
        Here is a list of what each choice contains:

        distance = 'very_close', 'close', 'sweet', 'far'
        rel_speed = 'slow', 'medium', 'fast'
        angle = 'left', 'ahead', 'fast'
        mine_distance = 'very_near', 'near', 'mid', 'far'
        mine_angle = 'left', 'ahead', 'fast'
        danger = 'imminent', 'risky', 'safe'
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
    """
    
    response = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )

    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(response.output[0].content[0].text + "\n")

    return response.output[0].content[0].text

def insert_gen_code(code: str):
    file_path = r"C:\Users\Orteg\OneDrive\Pictures\Documents\GitHub\Fuzzy-Comp\kessler-game\examples\fuzzy_aggressive_controller.py"

    # Read file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Modify content
    pattern = r"(?<=# BEGIN GENERATED CODE\n)(.*?)(?=\n# END GENERATED CODE)"
    new_content = re.sub(pattern, code, content, flags=re.DOTALL)

    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

