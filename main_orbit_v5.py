import math

CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
DEFAULT_MAX_SPEED = 6.0

# -------------------------
# Tunable parameters.
# These are intentionally grouped so evolver.py can mutate them later.
# -------------------------
PARAMS = {
    # Capture safety
    "neutral_margin": 3,
    "enemy_margin": 7,
    "defense_buffer": 4,

    # Garrison
    "base_garrison": 4.5,
    "garrison_prod_mul": 1.8,
    "home_extra_garrison_early": 3,

    # Target scoring
    "neutral_prod_weight_early": 190.0,
    "neutral_prod_weight_mid": 145.0,
    "neutral_cost_weight": 75.0,
    "enemy_prod_weight": 130.0,
    "enemy_cost_weight": 42.0,
    "enemy_swing_bonus": 10.0,
    "distance_bias": 13.0,

    # Opening
    "cheap_neutral_bonus": 22.0,
    "cheap_neutral_dist": 32.0,
    "cheap_neutral_ships": 13,

    # Penalties
    "eta_soft": 18.0,
    "eta_hard": 33.0,
    "late_eta_limit": 18.0,
    "comet_multiplier": 0.45,
    "already_assigned_penalty": 0.68,

    # Launch discipline
    "min_spare_to_attack_enemy": 8,
    "max_neutral_eta_early": 26.0,
    "max_neutral_eta_late": 18.0,
}

_GAME_TURN = 0
_LAST_RESET_SIGNATURE = None


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


def ray_circle_t(x, y, dx, dy, cx, cy, r):
    # dx,dy should be unit vector. Returns distance along ray.
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
    if t1 > 1e-7:
        return t1
    if t2 > 1e-7:
        return t2
    return None


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


def extract_comet_future(obs, comet_id, turns):
    """Best-effort comet prediction from obs['comets'].
    If the wrapper shape differs, returns None and caller falls back to current position.
    """
    groups = get_field(obs, "comets", []) or []
    for group in groups:
        ids = get_field(group, "planet_ids", None)
        paths = get_field(group, "paths", None)
        idx = get_field(group, "path_index", None)
        if ids is None or paths is None or idx is None:
            continue
        try:
            ids = list(ids)
            k = ids.index(comet_id)
            path = paths[k]
            j = int(idx + max(0, int(round(turns))))
            if 0 <= j < len(path):
                pt = path[j]
                return float(pt[0]), float(pt[1])
        except Exception:
            continue
    return None


def future_position(p, turns, angular_velocity, comet_ids, obs=None):
    if p.id in comet_ids and obs is not None:
        pt = extract_comet_future(obs, p.id, turns)
        if pt is not None:
            return pt
    return rotated_position(p, turns, angular_velocity, comet_ids)


def spawn_point(source, tx, ty):
    angle = math.atan2(ty - source.y, tx - source.x)
    return (
        source.x + math.cos(angle) * (source.radius + 0.11),
        source.y + math.sin(angle) * (source.radius + 0.11),
    )


def path_blocked(source, target, tx, ty, planets, comet_ids, angular_velocity, eta, obs=None):
    sx, sy = spawn_point(source, tx, ty)

    # Bounds are cheap sanity checks.
    if not (-2 <= tx <= BOARD_SIZE + 2 and -2 <= ty <= BOARD_SIZE + 2):
        return True

    # Sun collision is terminal. Use padding to avoid tangent suicide.
    if segment_circle_intersects(sx, sy, tx, ty, CENTER_X, CENTER_Y, SUN_RADIUS + 0.75):
        return True

    # Check blockers at current and sampled future positions.
    # This is a conservative approximation of the engine's swept movement.
    samples = (0.0, 0.25, 0.5, 0.75, 1.0) if eta > 4 else (0.0, 1.0)
    for p in planets:
        if p.id == source.id or p.id == target.id:
            continue
        pad = 0.38
        for frac in samples:
            t = eta * frac
            # Fleet position at this exact fraction of the flight
            fx = sx + frac * (tx - sx)
            fy = sy + frac * (ty - sy)
            
            if frac == 0.0:
                px, py = p.x, p.y
            else:
                px, py = future_position(p, t, angular_velocity, comet_ids, obs)
                
            # Point-to-point actual collision check at time t
            if dist_xy(fx, fy, px, py) <= p.radius + pad:
                return True

    return False


def first_hit_for_fleet(fleet, planets):
    dx = math.cos(fleet.angle)
    dy = math.sin(fleet.angle)

    best_t = float("inf")
    best_planet = None

    sun_t = ray_circle_t(fleet.x, fleet.y, dx, dy, CENTER_X, CENTER_Y, SUN_RADIUS)
    if sun_t is not None:
        best_t = sun_t

    # Out-of-bounds distance. If a planet is farther than exit point, ignore.
    boundary_t = distance_to_board_exit(fleet.x, fleet.y, dx, dy)
    if boundary_t is not None:
        best_t = min(best_t, boundary_t)

    for p in planets:
        t = ray_circle_t(fleet.x, fleet.y, dx, dy, p.x, p.y, p.radius)
        if t is not None and t < best_t:
            best_t = t
            best_planet = p

    if best_planet is None:
        return None, None
    eta = best_t / max(0.001, fleet_speed(fleet.ships))
    return best_planet, eta


def distance_to_board_exit(x, y, dx, dy):
    ts = []
    if dx > 1e-9:
        ts.append((BOARD_SIZE - x) / dx)
    elif dx < -1e-9:
        ts.append((0.0 - x) / dx)
    if dy > 1e-9:
        ts.append((BOARD_SIZE - y) / dy)
    elif dy < -1e-9:
        ts.append((0.0 - y) / dy)
    ts = [t for t in ts if t > 0]
    return min(ts) if ts else None


def build_arrivals(planets, fleets):
    """Return map: planet_id -> list of (owner, ships, eta)."""
    arr = {p.id: [] for p in planets}
    for f in fleets:
        hit, eta = first_hit_for_fleet(f, planets)
        if hit is not None:
            arr[hit.id].append((f.owner, f.ships, eta))
    for k in arr:
        arr[k].sort(key=lambda x: x[2])
    return arr


def incoming_ships(arrivals, planet_id, owner=None, not_owner=None, eta_limit=None):
    total = 0
    for o, ships, eta in arrivals.get(planet_id, []):
        if eta_limit is not None and eta > eta_limit:
            continue
        if owner is not None and o != owner:
            continue
        if not_owner is not None and o == not_owner:
            continue
        total += ships
    return total


def friendly_before(arrivals, planet_id, my_player, eta_limit):
    return incoming_ships(arrivals, planet_id, owner=my_player, eta_limit=eta_limit)


def enemy_before(arrivals, planet_id, my_player, eta_limit):
    return incoming_ships(arrivals, planet_id, not_owner=my_player, eta_limit=eta_limit)


def reserve_for_planet(p, my_player, arrivals, turn):
    base = PARAMS["base_garrison"] + PARAMS["garrison_prod_mul"] * max(1, p.production)
    if turn < 60:
        base += PARAMS["home_extra_garrison_early"]

    reserve = base

    # Keep enough current ships so known enemy arrivals do not flip us.
    enemy_events = [(o, s, e) for (o, s, e) in arrivals.get(p.id, []) if o != my_player]
    for _, ships, eta in enemy_events:
        friendly = friendly_before(arrivals, p.id, my_player, eta + 0.2)
        produced_before_hit = p.production * int(max(0.0, eta))
        needed_now = ships + PARAMS["defense_buffer"] - friendly - produced_before_hit
        if needed_now > reserve:
            reserve = needed_now

    return max(0, int(math.ceil(reserve)))


def estimate_capture(source, target, planned_to_target, max_speed, angular_velocity, comet_ids, arrivals, my_player, obs, turn):
    margin = PARAMS["neutral_margin"] if target.owner == -1 else PARAMS["enemy_margin"]
    required = max(1, target.ships + margin - planned_to_target)
    eta = 0.0
    tx, ty = target.x, target.y

    for _ in range(5):
        tx, ty = future_position(target, eta, angular_velocity, comet_ids, obs)
        d = dist_xy(source.x, source.y, tx, ty)
        speed = fleet_speed(required, max_speed)
        eta = d / max(0.001, speed)

        already_mine = planned_to_target + friendly_before(arrivals, target.id, my_player, eta + 0.2)

        if target.owner == -1:
            growth = 0
            # If enemy is already going to arrive before us, be conservative.
            enemy_pre = enemy_before(arrivals, target.id, my_player, eta + 0.2)
            total_needed = target.ships + enemy_pre + margin
        else:
            growth = target.production * int(eta + 1.0)
            # Reinforcements from the current owner matter most.
            owner_reinforce = incoming_ships(arrivals, target.id, owner=target.owner, eta_limit=eta + 0.2)
            total_needed = target.ships + growth + owner_reinforce + margin

        if already_mine >= total_needed:
            return 0, eta, tx, ty

        required = int(math.ceil(total_needed - already_mine))

    tx, ty = future_position(target, eta, angular_velocity, comet_ids, obs)
    return max(0, int(required)), eta, tx, ty


def phase(turn):
    if turn < 70:
        return "early"
    if turn < 360:
        return "mid"
    if turn < 455:
        return "late"
    return "end"


def target_score(source, target, required, eta, tx, ty, my_player, comet_ids, arrivals, turn):
    d = dist_xy(source.x, source.y, tx, ty)
    prod = max(1, target.production)
    ph = phase(turn)

    if target.owner == -1:
        prod_weight = PARAMS["neutral_prod_weight_early"] if ph == "early" else PARAMS["neutral_prod_weight_mid"]
        score = (prod * prod_weight) / (d + PARAMS["distance_bias"])
        score += (prod * PARAMS["neutral_cost_weight"]) / (required + 7.0)
        if target.ships <= PARAMS["cheap_neutral_ships"] and d <= PARAMS["cheap_neutral_dist"] and ph == "early":
            score += PARAMS["cheap_neutral_bonus"]
    else:
        score = (prod * PARAMS["enemy_prod_weight"]) / (d + 18.0)
        score += (prod * PARAMS["enemy_cost_weight"]) / (required + 10.0)
        score += PARAMS["enemy_swing_bonus"]

        # Opportunistic strike: enemy planet already emptied or under third-party pressure.
        if target.ships <= 12:
            score += 18.0
        non_mine_incoming = enemy_before(arrivals, target.id, my_player, eta + 2.0)
        if non_mine_incoming > 0:
            score += min(15.0, non_mine_incoming * 0.15)

    # Comets are temporary, low production; only take cheap/near ones.
    if target.id in comet_ids:
        score *= PARAMS["comet_multiplier"]
        if d > 20:
            score *= 0.35
        if required > 10:
            score *= 0.4

    # ETA penalties.
    if eta > PARAMS["eta_soft"]:
        score *= 0.75
    if eta > PARAMS["eta_hard"]:
        score *= 0.45

    # Endgame: long captures often do not repay before turn 500.
    turns_left = max(0, 500 - turn)
    if ph == "end":
        if eta > 12:
            score *= 0.2
        # In endgame, enemy ships/score swing matters more than new neutral production.
        if target.owner == -1:
            score *= 0.45
    elif ph == "late" and eta > PARAMS["late_eta_limit"]:
        score *= 0.5

    # Avoid draining tiny planets for marginal plays.
    if source.production <= 2 and required > source.ships * 0.65:
        score *= 0.72

    return score


def should_consider_target(source, target, eta, required, turn, comet_ids):
    ph = phase(turn)
    if target.id in comet_ids and (required > 10 or eta > 12):
        return False
    if target.owner == -1:
        if ph == "early" and eta > PARAMS["max_neutral_eta_early"]:
            return False
        if ph in ("late", "end") and eta > PARAMS["max_neutral_eta_late"]:
            return False
    else:
        if ph == "early":
            # Do not enemy-rush too early unless it is very close and cheap.
            if eta > 14 or required > 22:
                return False
    return True


def reset_turn_if_new_game(my_player, planets, fleets):
    global _GAME_TURN, _LAST_RESET_SIGNATURE
    my_planets = [p for p in planets if p.owner == my_player]
    # New games generally start with one owned planet, 10 ships, no fleets.
    sig = None
    if len(my_planets) == 1 and not fleets and my_planets[0].ships <= 12:
        sig = (my_player, my_planets[0].id, round(my_planets[0].x, 2), round(my_planets[0].y, 2), len(planets))
        if sig != _LAST_RESET_SIGNATURE:
            _GAME_TURN = 0
            _LAST_RESET_SIGNATURE = sig


def agent(obs, config=None):
    global _GAME_TURN

    my_player = int(get_field(obs, "player", 0))
    raw_planets = get_field(obs, "planets", []) or []
    raw_fleets = get_field(obs, "fleets", []) or []

    planets = [Planet(p) for p in raw_planets]
    fleets = [Fleet(f) for f in raw_fleets]
    if not planets:
        return []

    reset_turn_if_new_game(my_player, planets, fleets)
    turn = _GAME_TURN
    _GAME_TURN += 1

    angular_velocity = float(get_field(obs, "angular_velocity", 0.0) or 0.0)
    comet_ids = set(int(x) for x in (get_field(obs, "comet_planet_ids", []) or []))

    max_speed = DEFAULT_MAX_SPEED
    if config is not None:
        max_speed = float(get_field(config, "shipSpeed", DEFAULT_MAX_SPEED) or DEFAULT_MAX_SPEED)

    my_planets = [p for p in planets if p.owner == my_player]
    if not my_planets:
        return []

    arrivals = build_arrivals(planets, fleets)

    # Available ships after dynamic reserve.
    available = {}
    for p in my_planets:
        reserve = reserve_for_planet(p, my_player, arrivals, turn)
        available[p.id] = max(0, p.ships - reserve)

    moves = []
    planned_to_target = {p.id: friendly_before(arrivals, p.id, my_player, 999.0) for p in planets}

    # -------------------------
    # 1) Emergency defense with ETA-aware support.
    # -------------------------
    defense_jobs = []
    for target in my_planets:
        worst_deficit = 0
        for owner, ships, eta in arrivals.get(target.id, []):
            if owner == my_player:
                continue
            friendly = friendly_before(arrivals, target.id, my_player, eta + 0.2)
            produced = target.production * int(max(0, eta))
            deficit = ships + PARAMS["defense_buffer"] - target.ships - friendly - produced
            if deficit > worst_deficit:
                worst_deficit = deficit
        if worst_deficit > 0:
            defense_jobs.append((worst_deficit, target))

    defense_jobs.sort(key=lambda x: -x[0])

    for deficit, target in defense_jobs:
        helpers = sorted(
            [p for p in my_planets if p.id != target.id and available.get(p.id, 0) > 0],
            key=lambda s: dist_xy(s.x, s.y, target.x, target.y),
        )
        for source in helpers:
            if deficit <= 0:
                break
            send = min(available[source.id], int(math.ceil(deficit)))
            if send <= 0:
                continue
            tx, ty = future_position(target, 0, angular_velocity, comet_ids, obs)
            # Iterate target position by estimated ETA.
            for _ in range(3):
                eta = dist_xy(source.x, source.y, tx, ty) / max(0.001, fleet_speed(send, max_speed))
                tx, ty = future_position(target, eta, angular_velocity, comet_ids, obs)

            if path_blocked(source, target, tx, ty, planets, comet_ids, angular_velocity, eta, obs):
                continue

            angle = math.atan2(ty - source.y, tx - source.x)
            moves.append([source.id, angle, int(send)])
            available[source.id] -= send
            planned_to_target[target.id] = planned_to_target.get(target.id, 0) + send
            deficit -= send

    # -------------------------
    # 2) Expansion / attack.
    # Strong sources act first. One launch per source per turn keeps behavior stable.
    # -------------------------
    sources = sorted(my_planets, key=lambda p: (available.get(p.id, 0), p.production), reverse=True)

    for source in sources:
        spare = available.get(source.id, 0)
        if spare <= 0:
            continue

        best = None  # (score, target, required, eta, tx, ty)

        for target in planets:
            if target.id == source.id or target.owner == my_player:
                continue

            already = planned_to_target.get(target.id, 0)

            required, eta, tx, ty = estimate_capture(
                source, target, already, max_speed, angular_velocity, comet_ids,
                arrivals, my_player, obs, turn
            )

            if required <= 0 or required > spare:
                continue

            if target.owner != -1 and spare < PARAMS["min_spare_to_attack_enemy"]:
                continue

            if not should_consider_target(source, target, eta, required, turn, comet_ids):
                continue

            if source.ships - required < 3 and target.owner != -1:
                continue

            if path_blocked(source, target, tx, ty, planets, comet_ids, angular_velocity, eta, obs):
                continue

            score = target_score(source, target, required, eta, tx, ty, my_player, comet_ids, arrivals, turn)

            if already > 0:
                score *= PARAMS["already_assigned_penalty"]

            # Prefer targets that increase local control: close to our source but not too near the sun.
            radial = dist_xy(tx, ty, CENTER_X, CENTER_Y)
            if radial > SUN_RADIUS + 8:
                score *= 1.03

            # Avoid investing too much in a low-prod neutral after opening.
            if target.owner == -1 and phase(turn) != "early" and target.production <= 2 and required > 18:
                score *= 0.55

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
