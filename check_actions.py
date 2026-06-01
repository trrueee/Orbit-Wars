import json

with open("episode-78412298-replay.json", "r") as f:
    data = json.load(f)

steps = data["steps"]

print("=== Actions in the first 60 steps ===")
for i, step_data in enumerate(steps[:60]):
    p0_action = step_data[0].get("action")
    p1_action = step_data[1].get("action")
    
    if p0_action or p1_action:
        print(f"Step {i}:")
        if p0_action:
            print(f"  P0 (Us)      : {p0_action}")
        if p1_action:
            print(f"  P1 (Opponent): {p1_action}")
