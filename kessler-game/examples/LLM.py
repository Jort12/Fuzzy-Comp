from openai import OpenAI
client = OpenAI(api_key="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 ")

prompt = """
Create a different Fuzzy Rule Set for an Aggresive Play Style use this style:
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
    model = 'chatgpt-4o-latest',
    imput = prompt
)

