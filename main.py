import math

CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
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
    """Orbit Wars speed formula: small fleets are slow; large fleets approach max_speed."""
    ships = max(1, int(ships))
    if ships <= 1:
        return 1.0
    ratio = math.log(min(ships, 1000)) / math.log(1000)
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)


def ray_circle_t(x, y, dx, dy, cx, cy, r):
    """Return first positive distance t along a unit ray, or None if no hit."""
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
    """Predict ordinary orbiting planets from their current position.
    Comets are deliberately treated as current-position targets because their path data format
    can differ between wrappers; attacking only close comets keeps this safe.
    """
    if not is_orbiting_planet(p, comet_ids):
        return p.x, p.y
    vx = p.x - CENTER_X
    vy = p.y - CENTER_Y
    a = angular_velocity * turns
    ca = math.cos(a)
    sa = math.sin(a)
    return CENTER_X + vx * ca - vy * sa, CENTER_Y + vx * sa + vy * ca


def path_blocked(source, target, tx, ty, planets, comet_ids, angular_velocity, eta):
    # Avoid the sun. Padding prevents near-tangent suicide routes.
    if segment_circle_intersects(source.x, source.y, tx, ty, CENTER_X, CENTER_Y, SUN_RADIUS + 0.9):
        return True

    # Avoid immediate/current blockers and likely future blockers.
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

    # If the fleet will hit the sun first, it should not be counted as a planet threat.
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
    """Return (required_ships, eta, target_x, target_y).
    The required amount is intentionally a little conservative. It accounts for enemy production
    during travel and uses the real size-dependent fleet speed formula.
    """
    margin = 3 if target.owner == -1 else 6
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
            # Enemy planets produce while our fleet is travelling. Ceil-ish without importing.
            growth = target.production * int(eta + 1.0)

        required = max(1, target.ships + growth + margin - already_assigned)

    tx, ty = rotated_position(target, eta, angular_velocity, comet_ids)
    return required, eta, tx, ty


def target_score(source, target, required, eta, tx, ty, my_player, comet_ids):
    d = dist_xy(source.x, source.y, tx, ty)
    prod = max(1, target.production)

    # Expansion first: production is king, but expensive/far planets should wait.
    if target.owner == -1:
        score = (prod * 150.0) / (d + 12.0) + (prod * 55.0) / (required + 6.0)
        if required <= 12:
            score += 12.0
    else:
        # Enemy captures matter, but don't suicide into distant fortified planets.
        score = (prod * 115.0) / (d + 18.0) + (prod * 40.0) / (required + 10.0)
        score += 10.0  # conquest swing bonus

    # Comets are temporary and production 1; only take them when very convenient.
    if target.id in comet_ids:
        score *= 0.55
        if d > 22.0:
            score *= 0.25

    # Prefer near-term captures; very long flights are often stale because planets rotate.
    if eta > 18:
        score *= 0.75
    if eta > 30:
        score *= 0.55

    return score


def safe_garrison(p, enemy_incoming, friendly_incoming):
    # Less conservative than always keeping 10 everywhere, but still guards valuable planets.
    base = 5 + 2 * max(1, p.production)
    need_vs_attack = enemy_incoming - friendly_incoming + 3
    return max(base, need_vs_attack)


def agent(obs, config=None):
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

    # Current known friendly fleets already heading into each target.
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

    # 1) Emergency defense. Save planets that are likely to flip.
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

    # 2) Expansion / attack. One high-confidence launch per source per turn.
    # Process stronger sources first so big planets take the best opportunities.
    sources = sorted(my_planets, key=lambda p: available.get(p.id, 0), reverse=True)

    for source in sources:
        spare = available.get(source.id, 0)
        if spare <= 0:
            continue

        best = None  # (score, target, required, eta, tx, ty)

        for target in planets:
            if target.id == source.id or target.owner == my_player:
                continue

            # Don't chase expensive comets; they disappear and only produce 1.
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

            # Avoid all-in attacks from small planets unless the target is very cheap.
            if source.ships - required < 4 and target.owner != -1:
                continue

            if path_blocked(source, target, tx, ty, planets, comet_ids, angular_velocity, eta):
                continue

            score = target_score(source, target, required, eta, tx, ty, my_player, comet_ids)

            # Slight preference for targets not already assigned this turn.
            if already > 0:
                score *= 0.72

            # Very cheap neutrals are excellent opening snowball targets.
            if target.owner == -1 and target.ships <= 12 and dist_xy(source.x, source.y, tx, ty) < 35:
                score += 18.0

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
