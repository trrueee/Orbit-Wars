import sys
import os
import argparse
import importlib.util
import math
import logging

# Silence TensorFlow and logger noises
logging.disable(logging.CRITICAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from kaggle_environments import make

# Add current path to sys.path so we can import local modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def load_agent(filepath):
    """Dynamically load an agent function from a python file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Agent file not found: {filepath}")
    spec = importlib.util.spec_from_file_location("dynamic_agent", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent

def run_single_game(args):
    """Run a single game episode and extract advanced metrics."""
    seed, agent_p0, agent_p1, agent_p2, agent_p3 = args
    env = make("orbit_wars", debug=False)
    
    # Run the match
    env.run([agent_p0, agent_p1, agent_p2, agent_p3])
    
    steps = env.steps
    last_step = steps[-1]
    
    # Determine standings and metrics
    rewards = [p.get("reward", -1.0) for p in last_step]
    statuses = [p.get("status", "DONE") for p in last_step]
    
    p0_reward = rewards[0]
    p0_status = statuses[0]
    is_win = (p0_reward == 1.0)
    is_error = (p0_status not in ["DONE", "ACTIVE"])
    
    # Track final planet and ship count
    final_obs = last_step[0]["observation"]
    planets = final_obs.get("planets", [])
    
    p0_planets = sum(1 for p in planets if p[1] == 0)
    p0_ships = sum(p[5] for p in planets if p[1] == 0)
    
    # Find when we captured our first planet
    first_capture_turn = None
    for turn_idx, step_data in enumerate(steps):
        obs = step_data[0]["observation"]
        owned_planets = sum(1 for p in obs.get("planets", []) if p[1] == 0)
        if owned_planets >= 2:
            first_capture_turn = turn_idx
            break
            
    if first_capture_turn is None:
        first_capture_turn = len(steps)
        
    return {
        "is_win": is_win,
        "is_error": is_error,
        "reward": p0_reward,
        "planets": p0_planets,
        "ships": p0_ships,
        "first_capture_turn": first_capture_turn,
        "steps": len(steps)
    }

def main():
    parser = argparse.ArgumentParser(description="Orbit Wars Local Benchmark Arena")
    parser.add_argument("--games", type=int, default=20, help="Number of matches to run (default: 20)")
    parser.add_argument("--opponent", type=str, default="opponents/bot_v1.py", help="Opponent agent filepath (default: opponents/bot_v1.py)")
    parser.add_argument("--mode", type=str, choices=["all_random", "one_vs_one", "all_opponent"], default="one_vs_one",
                        help="Matchup mode: all_random (1 vs 3 randoms), one_vs_one (1 vs 1 opp + 2 randoms), all_opponent (1 vs 3 opps)")
    parser.add_argument("--start-seed", type=int, default=1000, help="Starting random seed value (default: 1000)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("      *** WELCOME TO ORBIT WARS LOCAL TRAINING ARENA ***      ")
    print("="*60)
    
    # Warm up environment in main thread
    print("[*] Warming up Orbit Wars environment...", end="")
    try:
        warmup_env = make("orbit_wars", debug=False)
        print(" [OK]")
    except Exception as e:
        print(f" [FAILED]\nError: {e}")
        sys.exit(1)
    
    # Load main agent (d:\Project\bot\main.py)
    print("[*] Loading Current Agent (main.py)...", end="")
    try:
        current_agent = load_agent("main.py")
        print(" [OK]")
    except Exception as e:
        print(f" [FAILED]\nError: {e}")
        sys.exit(1)
        
    # Load opponent agent
    opp_agent = "random"
    if args.opponent != "random":
        print(f"[*] Loading Sparring Opponent ({args.opponent})...", end="")
        try:
            opp_agent = load_agent(args.opponent)
            print(" [OK]")
        except Exception as e:
            print(f" [FAILED]\nError: {e}")
            sys.exit(1)
            
    # Set up player slot assignments
    agent_p0 = current_agent
    if args.mode == "all_random":
        agent_p1 = "random"
        agent_p2 = "random"
        agent_p3 = "random"
        opp_name = "random"
    elif args.mode == "one_vs_one":
        agent_p1 = opp_agent
        agent_p2 = "random"
        agent_p3 = "random"
        opp_name = f"{args.opponent} (1) & random (2)"
    else:  # all_opponent
        agent_p1 = opp_agent
        agent_p2 = opp_agent
        agent_p3 = opp_agent
        opp_name = f"{args.opponent} (3)"
        
    print(f"[MODE] Simulation Mode: {args.mode.upper()}")
    print(f"[LAYOUT] Opponent Layout: Player 0 (Current) vs. {opp_name}")
    print(f"[RUN] Running {args.games} games sequentially (100% Safe Mode)...\n")
    
    # Prepare argument payloads
    tasks = []
    for idx in range(args.games):
        seed = args.start_seed + idx
        tasks.append((seed, agent_p0, agent_p1, agent_p2, agent_p3))
        
    # Execute batch matches sequentially
    results = []
    completed = 0
    
    for task in tasks:
        res = run_single_game(task)
        results.append(res)
        completed += 1
        sys.stdout.write(f"\rProgress: [{completed}/{args.games}] games simulated...")
        sys.stdout.flush()
            
    print("\n" + "-"*60)
    print("--- BENCHMARK METRICS SUMMARY:")
    print("-"*60)
    
    # Aggregate data
    total_wins = sum(1 for r in results if r["is_win"])
    total_errors = sum(1 for r in results if r["is_error"])
    avg_reward = sum(r["reward"] for r in results) / args.games
    avg_planets = sum(r["planets"] for r in results) / args.games
    avg_ships = sum(r["ships"] for r in results) / args.games
    avg_first_cap = sum(r["first_capture_turn"] for r in results) / args.games
    avg_steps = sum(r["steps"] for r in results) / args.games
    
    win_rate = (total_wins / args.games) * 100
    error_rate = (total_errors / args.games) * 100
    
    # Print metrics
    print(f"Win Rate (1st Place)         : {win_rate:.1f}% ({total_wins}/{args.games})")
    print(f"Avg Leaderboard Reward        : {avg_reward:+.3f}  [-1.0 to +1.0]")
    print(f"Avg Final Planet Ownership   : {avg_planets:.1f} planets")
    print(f"Avg Final Garrison Fleet     : {avg_ships:.1f} ships")
    print(f"Avg First Capture Turn       : step {avg_first_cap:.1f}  (lower is faster)")
    print(f"Avg Match Steps Run          : {avg_steps:.1f} steps")
    print(f"Crash / Timeout Rate        : {error_rate:.1f}% ({total_errors}/{args.games})")
    print("="*60)
    
    if win_rate >= 80.0:
        print("[RECOMMENDATION] Extremely dominant! Safe to submit to Kaggle!")
    elif win_rate >= 50.0:
        print("[RECOMMENDATION] Solid improvement. Decent submission candidate.")
    else:
        print("[RECOMMENDATION] Strategy needs tuning. Underperforming sparring partner.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
