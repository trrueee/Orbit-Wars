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

def path_blocked(source, target, tx, ty, planets):
    if segment_circle_intersects(source.x, source.y, tx, ty, CENTER_X, CENTER_Y, SUN_RADIUS + 0.5):
        return True
    for p in planets:
        if p.id == source.id or p.id == target.id:
            continue
        if segment_circle_intersects(source.x, source.y, tx, ty, p.x, p.y, p.radius + 0.3):
            return True
    return False

def agent(obs, config=None):
    my_player = int(get_field(obs, "player", 0))
    raw_planets = get_field(obs, "planets", []) or []
    planets = [Planet(p) for p in raw_planets]
    
    if not planets:
        return []
        
    my_planets = [p for p in planets if p.owner == my_player]
    if not my_planets:
        return []
        
    # Rush Bot: Very low garrison threshold (just keeps 2 + 1*prod ships)
    available = {}
    for p in my_planets:
        reserve = 2 + p.production
        available[p.id] = max(0, p.ships - reserve)
        
    moves = []
    
    # Fast launch towards closest cheap neutral first
    for source in sorted(my_planets, key=lambda p: available.get(p.id, 0), reverse=True):
        spare = available.get(source.id, 0)
        if spare <= 5: # Needs minimal force to attack
            continue
            
        best = None
        for target in planets:
            if target.id == source.id or target.owner == my_player:
                continue
                
            d = dist_xy(source.x, source.y, target.x, target.y)
            required = target.ships + (2 if target.owner == -1 else 4)
            
            if required > spare:
                continue
                
            if path_blocked(source, target, target.x, target.y, planets):
                continue
                
            # Rush scoring formula: extreme priority to proximity and cheapness
            if target.owner == -1:
                score = 300.0 / (d + 2.0) + 100.0 / (required + 1.0)
            else:
                score = 150.0 / (d + 5.0) + 50.0 / (required + 1.0)
                
            if best is None or score > best[0]:
                best = (score, target, required)
                
        if best is not None:
            _, target, required = best
            angle = math.atan2(target.y - source.y, target.x - source.x)
            moves.append([source.id, angle, int(required)])
            available[source.id] -= required
            
    return moves
