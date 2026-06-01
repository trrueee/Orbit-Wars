import sys
import os
import argparse
import importlib.util
import math
import logging
import random

# Silence loggers
logging.disable(logging.CRITICAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from kaggle_environments import make

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# -------------------------------------------------------------
# GLOBAL PARAMETERS DICTIONARY (MUTABLE BY CANDIDATE)
# -------------------------------------------------------------
P = {
    "margin_neutral": 3.0,
    "margin_enemy": 6.0,
    "neutral_dist_mult": 150.0,
    "neutral_req_mult": 55.0,
    "enemy_dist_mult": 115.0,
    "enemy_req_mult": 40.0,
    "enemy_conquest_bonus": 10.0,
    "garrison_base": 5.0,
    "garrison_prod_mult": 2.0,
    "cheap_opening_dist": 35.0,
    "cheap_opening_bonus": 18.0
}

# -------------------------------------------------------------
# TOURNAMENT BOT ABSTRACTED IMPLEMENTATION
# -------------------------------------------------------------
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
DEFAULT_MAX_SPEED = 6.0

class Planet:
    __slots__ = ("id", "owner", "x", "y", "radius", "ships", "production")
    def __init__(self, raw):
        self.id = int(raw[0])
        self.owner = int(raw[1])
        self.x = float(raw[2])
        self.y = float(raw[3])
        self.radius = float(raw[4])
        self.ships = int(raw[5])
        self.production = int(raw[6])

class Fleet:
    __slots__ = ("id", "owner", "x", "y", "angle", "from_planet_id", "ships")
    def __init__(self, raw):
        self.id = int(raw[0])
        self.owner = int(raw[1])
        self.x = float(raw[2])
        self.y = float(raw[3])
        self.angle = float(raw[4])
        self.from_planet_id = int(raw[5])
        self.ships = int(raw[6])

def get_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def dist_xy(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def fleet_speed(ships, max_speed=DEFAULT_MAX_SPEED):
    ships = max(1, int(ships))
    if ships <= 1:
        return 1.0
    ratio = math.log(min(ships, 1000)) / math.log(1000)
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)

def ray_circle_t(x, y, dx, dy, cx, cy, r):
    ox = x - cx
    oy = y - cy
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - r * r
    disc = b * b - 4.0 * c
    if disc < 0:
        return None
    root = math.sqrt(disc)
    t1 = (-b - root) / 2.0
    t2 = (-b + root) / 2.0
    if t1 > 0:
        return t1
    if t2 > 0:
        return t2
    return None

def segment_circle_intersects(x1, y1, x2, y2, cx, cy, r):
    vx = x2 - x1
    vy = y2 - y1
    wx = cx - x1
    wy = cy - y1
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= 1e-9:
        return wx * wx + wy * wy <= r * r
    t = (wx * vx + wy * vy) / seg_len_sq
    t = clamp(t, 0.0, 1.0)
    qx = x1 + t * vx
    qy = y1 + t * vy
    return (cx - qx) ** 2 + (cy - qy) ** 2 <= r * r

def is_orbiting_planet(p, comet_ids):
    if p.id in comet_ids:
        return False
    orbital_r = dist_xy(p.x, p.y, CENTER_X, CENTER_Y)
    return orbital_r + p.radius < 50.0

def rotated_position(p, turns, angular_velocity, comet_ids):
    if not is_orbiting_planet(p, comet_ids):
        return p.x, p.y
    vx = p.x - CENTER_X
    vy = p.y - CENTER_Y
    a = angular_velocity * turns
    ca = math.cos(a)
    sa = math.sin(a)
    return CENTER_X + vx * ca - vy * sa, CENTER_Y + vx * sa + vy * ca

def path_blocked(source, target, tx, ty, planets, comet_ids, angular_velocity, eta):
    if segment_circle_intersects(source.x, source.y, tx, ty, CENTER_X, CENTER_Y, SUN_RADIUS + 0.9):
        return True
    for p in planets:
        if p.id == source.id or p.id == target.id:
            continue
        pad = 0.45
        if segment_circle_intersects(source.x, source.y, tx, ty, p.x, p.y, p.radius + pad):
            return True
        if eta > 3 and is_orbiting_planet(p, comet_ids):
            px, py = rotated_position(p, eta, angular_velocity, comet_ids)
            if segment_circle_intersects(source.x, source.y, tx, ty, px, py, p.radius + pad):
                return True
    return False

def first_planet_on_fleet_ray(fleet, planets):
    dx = math.cos(fleet.angle)
    dy = math.sin(fleet.angle)
    best_t = float("inf")
    best_planet = None
    sun_t = ray_circle_t(fleet.x, fleet.y, dx, dy, CENTER_X, CENTER_Y, SUN_RADIUS)
    if sun_t is not None:
        best_t = sun_t
    for p in planets:
        t = ray_circle_t(fleet.x, fleet.y, dx, dy, p.x, p.y, p.radius)
        if t is not None and t < best_t:
            best_t = t
            best_planet = p
    return best_planet

def incoming_by_owner(planets, fleets):
    incoming = {}
    for f in fleets:
        hit = first_planet_on_fleet_ray(f, planets)
        if hit is None:
            continue
        key = (f.owner, hit.id)
        incoming[key] = incoming.get(key, 0) + f.ships
    return incoming

def estimate_capture(source, target, already_assigned, max_speed, angular_velocity, comet_ids):
    margin = P["margin_neutral"] if target.owner == -1 else P["margin_enemy"]
    required = max(1, target.ships + margin - already_assigned)
    eta = 0.0
    tx, ty = target.x, target.y
    for _ in range(4):
        tx, ty = rotated_position(target, eta, angular_velocity, comet_ids)
        d = dist_xy(source.x, source.y, tx, ty)
        speed = fleet_speed(required, max_speed)
        eta = d / max(speed, 0.001)
        if target.owner == -1:
            growth = 0
        else:
            growth = target.production * int(eta + 1.0)
        required = max(1, target.ships + growth + margin - already_assigned)
    tx, ty = rotated_position(target, eta, angular_velocity, comet_ids)
    return required, eta, tx, ty

def target_score(source, target, required, eta, tx, ty, my_player, comet_ids):
    d = dist_xy(source.x, source.y, tx, ty)
    prod = max(1, target.production)
    if target.owner == -1:
        score = (prod * P["neutral_dist_mult"]) / (d + 12.0) + (prod * P["neutral_req_mult"]) / (required + 6.0)
        if required <= 12:
            score += 12.0
    else:
        score = (prod * P["enemy_dist_mult"]) / (d + 18.0) + (prod * P["enemy_req_mult"]) / (required + 10.0)
        score += P["enemy_conquest_bonus"]
    if target.id in comet_ids:
        score *= 0.55
        if d > 22.0:
            score *= 0.25
    if eta > 18:
        score *= 0.75
    if eta > 30:
        score *= 0.55
    return score

def safe_garrison(p, enemy_incoming, friendly_incoming):
    base = P["garrison_base"] + P["garrison_prod_mult"] * max(1, p.production)
    need_vs_attack = enemy_incoming - friendly_incoming + 3
    return max(base, need_vs_attack)

def candidate_agent(obs, config=None):
    my_player = int(get_field(obs, "player", 0))
    raw_planets = get_field(obs, "planets", []) or []
    raw_fleets = get_field(obs, "fleets", []) or []
    planets = [Planet(p) for p in raw_planets]
    fleets = [Fleet(f) for f in raw_fleets]
    if not planets:
        return []
    angular_velocity = float(get_field(obs, "angular_velocity", 0.0) or 0.0)
    comet_ids = set(int(x) for x in (get_field(obs, "comet_planet_ids", []) or []))
    max_speed = DEFAULT_MAX_SPEED
    if config is not None:
        max_speed = float(get_field(config, "shipSpeed", DEFAULT_MAX_SPEED) or DEFAULT_MAX_SPEED)
    incoming = incoming_by_owner(planets, fleets)
    my_planets = [p for p in planets if p.owner == my_player]
    if not my_planets:
        return []
    friendly_incoming = {p.id: incoming.get((my_player, p.id), 0) for p in planets}
    enemy_incoming = {}
    for p in planets:
        total = 0
        for owner in range(4):
            if owner != my_player:
                total += incoming.get((owner, p.id), 0)
        enemy_incoming[p.id] = total
    available = {}
    for p in my_planets:
        reserve = safe_garrison(p, enemy_incoming.get(p.id, 0), friendly_incoming.get(p.id, 0))
        available[p.id] = max(0, p.ships - reserve)
    moves = []
    planned_to_target = {p.id: friendly_incoming.get(p.id, 0) for p in planets}
    needy = []
    for p in my_planets:
        deficit = enemy_incoming.get(p.id, 0) + 3 - p.ships - friendly_incoming.get(p.id, 0)
        if deficit > 0:
            needy.append((deficit, p))
    needy.sort(key=lambda x: -x[0])
    for deficit, target in needy:
        if deficit <= 0:
            continue
        helpers = sorted(
            [p for p in my_planets if p.id != target.id and available.get(p.id, 0) > 0],
            key=lambda s: dist_xy(s.x, s.y, target.x, target.y),
        )
        for source in helpers:
            if deficit <= 0:
                break
            send = min(available[source.id], deficit)
            if send <= 0:
                continue
            eta = dist_xy(source.x, source.y, target.x, target.y) / fleet_speed(send, max_speed)
            tx, ty = rotated_position(target, eta, angular_velocity, comet_ids)
            if path_blocked(source, target, tx, ty, planets, comet_ids, angular_velocity, eta):
                continue
            angle = math.atan2(ty - source.y, tx - source.x)
            moves.append([source.id, angle, int(send)])
            available[source.id] -= send
            planned_to_target[target.id] = planned_to_target.get(target.id, 0) + send
            deficit -= send
    sources = sorted(my_planets, key=lambda p: available.get(p.id, 0), reverse=True)
    for source in sources:
        spare = available.get(source.id, 0)
        if spare <= 0:
            continue
        best = None
        for target in planets:
            if target.id == source.id or target.owner == my_player:
                continue
            if target.id in comet_ids and target.ships > 10:
                continue
            already = planned_to_target.get(target.id, 0)
            required, eta, tx, ty = estimate_capture(
                source, target, already, max_speed, angular_velocity, comet_ids
            )
            if required <= 0:
                continue
            if required > spare:
                continue
            if source.ships - required < 4 and target.owner != -1:
                continue
            if path_blocked(source, target, tx, ty, planets, comet_ids, angular_velocity, eta):
                continue
            score = target_score(source, target, required, eta, tx, ty, my_player, comet_ids)
            if already > 0:
                score *= 0.72
            if target.owner == -1 and target.ships <= 12 and dist_xy(source.x, source.y, tx, ty) < P["cheap_opening_dist"]:
                score += P["cheap_opening_bonus"]
            if best is None or score > best[0]:
                best = (score, target, required, eta, tx, ty)
        if best is not None:
            _, target, required, eta, tx, ty = best
            send = int(min(spare, required))
            if send > 0:
                angle = math.atan2(ty - source.y, tx - source.x)
                moves.append([source.id, angle, send])
                available[source.id] -= send
                planned_to_target[target.id] = planned_to_target.get(target.id, 0) + send
    return moves

# -------------------------------------------------------------
# DYNAMIC LOADER FOR OPPONENTS
# -------------------------------------------------------------
def load_agent(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Sparring partner not found: {filepath}")
    spec = importlib.util.spec_from_file_location("dynamic_opp", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent

# -------------------------------------------------------------
# EVOLUTION ENGINE CORE
# -------------------------------------------------------------
def mutate_params(parent):
    child = {}
    for k, v in parent.items():
        if k in ["neutral_dist_mult", "neutral_req_mult", "enemy_dist_mult", "enemy_req_mult", "enemy_conquest_bonus", "cheap_opening_bonus"]:
            noise = random.gauss(0, 10.0)
            child[k] = round(clamp(v + noise, 30.0, 240.0), 1)
        elif k in ["cheap_opening_dist"]:
            noise = random.gauss(0, 2.5)
            child[k] = round(clamp(v + noise, 20.0, 48.0), 1)
        elif k in ["margin_neutral", "margin_enemy", "garrison_base", "garrison_prod_mult"]:
            noise = random.gauss(0, 0.8)
            child[k] = round(clamp(v + noise, 1.0, 10.0), 1)
        else:
            child[k] = v
    return child

def evaluate_candidate(candidate_name, candidate_params, opp_v3, opp_rush, opp_turtle, opp_baseline, start_seed=1000, num_games=8):
    global P
    P.update(candidate_params)
    
    wins = 0
    total_reward = 0.0
    total_planets = 0
    total_ships = 0
    
    # 4 games of 1v1 against bot_v3 (to test elite capability)
    # 4 games of chaos mixed-league (to test general robustness)
    for idx in range(num_games):
        seed = start_seed + idx
        env = make("orbit_wars", debug=False)
        
        if idx < num_games // 2:
            # Track 1: 1v1 against bot_v3
            env.run([candidate_agent, opp_v3, "random", "random"])
        else:
            # Track 2: Chaos 4-Player League
            env.run([candidate_agent, opp_v3, opp_rush, opp_turtle])
            
        last_step = env.steps[-1]
        p0_reward = last_step[0].get("reward", -1.0)
        
        if p0_reward == 1.0:
            wins += 1
        total_reward += p0_reward
        
        final_obs = last_step[0]["observation"]
        planets = final_obs.get("planets", [])
        p0_planets = sum(1 for p in planets if p[1] == 0)
        p0_ships = sum(p[5] for p in planets if p[1] == 0)
        
        total_planets += p0_planets
        total_ships += p0_ships

    win_rate = (wins / num_games) * 100
    avg_reward = total_reward / num_games
    avg_planets = total_planets / num_games
    avg_ships = total_ships / num_games
    
    # Fitness Function incorporating win rate, reward and size metrics to prevent mixing
    fitness = win_rate * 10.0 + avg_reward * 200.0 + avg_planets * 10.0 + avg_ships * 0.05
    return {
        "name": candidate_name,
        "win_rate": win_rate,
        "reward": avg_reward,
        "planets": avg_planets,
        "ships": avg_ships,
        "fitness": fitness,
        "params": candidate_params.copy()
    }

def main():
    parser = argparse.ArgumentParser(description="Self-Play Genetic Parameter Evolution League")
    parser.add_argument("--generations", type=int, default=3, help="Number of genetic search generations (default: 3)")
    parser.add_argument("--candidates-per-gen", type=int, default=3, help="Number of mutated offspring per generation (default: 3)")
    parser.add_argument("--games-per-candidate", type=int, default=8, help="Matches per candidate (Track 1 + 2 combined, default: 8)")
    args = parser.parse_args()
    
    print("\n" + "="*65)
    print("    *** SELF-PLAY GENETIC LEAGUE ENGINE (v3 CHAMP SPARRING) ***   ")
    print("="*65)
    
    # Load opponents
    print("[*] Warming up and loading sparring pool...")
    try:
        opp_v3 = load_agent("opponents/bot_v3.py")
        print("    -> [OK] bot_v3.py (625-Point Master Sparrer - Baseline)")
        opp_rush = load_agent("opponents/bot_rush.py")
        print("    -> [OK] bot_rush.py (Rush Sparrer)")
        opp_turtle = load_agent("opponents/bot_turtle.py")
        print("    -> [OK] bot_turtle.py (Turtle Sparrer)")
        opp_baseline = load_agent("opponents/bot_v1.py")
        print("    -> [OK] bot_v1.py (Baseline Heuristics)")
    except Exception as e:
        print(f" [FAILED] Error loading pool: {e}")
        sys.exit(1)
        
    # Start seed parent (v3 parameters)
    parent_params = {
        "margin_neutral": 3.0,
        "margin_enemy": 6.0,
        "neutral_dist_mult": 150.0,
        "neutral_req_mult": 55.0,
        "enemy_dist_mult": 115.0,
        "enemy_req_mult": 40.0,
        "enemy_conquest_bonus": 10.0,
        "garrison_base": 5.0,
        "garrison_prod_mult": 2.0,
        "cheap_opening_dist": 35.0,
        "cheap_opening_bonus": 18.0
    }
    
    print("\n[*] Evaluating Parent 0 (v3 Baseline)...", end="")
    parent_metrics = evaluate_candidate(
        "v3_parent", parent_params, opp_v3, opp_rush, opp_turtle, opp_baseline, start_seed=4000, num_games=args.games_per_candidate
    )
    print(" [OK]")
    print(f"    -> WinRate: {parent_metrics['win_rate']:.1f}%, AvgReward: {parent_metrics['reward']:+.3f}, Fitness: {parent_metrics['fitness']:.1f}")
    
    current_parent = parent_metrics
    history = [current_parent]
    
    # Generation Loop
    for g in range(1, args.generations + 1):
        print("\n" + "-"*65)
        print(f"=== GENERATION {g} / {args.generations} (Evolving from best parent) ===")
        print("-"*65)
        
        candidates = []
        # Always evaluate the parent to maintain exact seed alignment and avoid stochastic drift
        candidates.append(current_parent)
        
        for c in range(1, args.candidates_per_gen + 1):
            c_name = f"Gen{g}_Child{c}"
            c_params = mutate_params(current_parent["params"])
            print(f"[*] Evaluating candidate {c_name}...", end="")
            sys.stdout.flush()
            
            res = evaluate_candidate(
                c_name, c_params, opp_v3, opp_rush, opp_turtle, opp_baseline, start_seed=4000+g*100, num_games=args.games_per_candidate
            )
            print(" [OK]")
            print(f"    ├─ Params: CheapOpen(dist={c_params['cheap_opening_dist']}, bonus={c_params['cheap_opening_bonus']}), Garrison({c_params['garrison_base']}+{c_params['garrison_prod_mult']}*P)")
            print(f"    └─ WinRate: {res['win_rate']:.1f}%, AvgReward: {res['reward']:+.3f}, Fitness: {res['fitness']:.1f}")
            candidates.append(res)
            
        # Select best candidate of this generation
        candidates.sort(key=lambda x: -x["fitness"])
        best_candidate = candidates[0]
        
        if best_candidate["fitness"] > current_parent["fitness"]:
            print(f"\n[EVOLUTION SUCCESS] Candidate '{best_candidate['name']}' defeated the parent!")
            print(f"    └─ Fitness Improvement: {current_parent['fitness']:.1f} -> {best_candidate['fitness']:.1f}")
            current_parent = best_candidate
            history.append(current_parent)
        else:
            print("\n[EVOLUTION STAGNATION] Parent remains the dominant strategist.")
            
    # -------------------------------------------------------------
    # VALIDATION / VERIFICATION PHASE (ANTI-OVERFITTING)
    # -------------------------------------------------------------
    print("\n" + "="*65)
    print("      *** VERIFICATION PHASE (ANTI-OVERFITTING SHIELD) ***      ")
    print("="*65)
    print("[*] Simulating final champion on UNKNOWN SEEDS against v3...")
    
    # 10 matches on unseen seeds [8000-8009]
    validation_metrics = evaluate_candidate(
        "validation_champion", current_parent["params"], opp_v3, opp_rush, opp_turtle, opp_baseline, start_seed=8000, num_games=10
    )
    
    print("\n=== FINAL VALIDATION RESULTS (vs v3 Champion on Unseen Seeds):")
    print(f"    └─ Final Win Rate vs. v3    : {validation_metrics['win_rate']:.1f}%")
    print(f"    └─ Avg Leaderboard Reward   : {validation_metrics['reward']:+.3f}")
    print(f"    └─ Avg Final Planet Count   : {validation_metrics['planets']:.1f}")
    print(f"    └─ Avg Final Garrison Fleet : {validation_metrics['ships']:.1f}")
    
    print("\n=== OPTIMIZED CHAMPION SUPER-PARAMETERS:")
    for k, v in current_parent["params"].items():
        print(f"    -> {k:<24} : {v}")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
