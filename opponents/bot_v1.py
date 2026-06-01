import math

class Planet:
    def __init__(self, raw):
        self.id = int(raw[0])
        self.owner = int(raw[1])
        self.x = float(raw[2])
        self.y = float(raw[3])
        self.radius = float(raw[4])
        self.ships = int(raw[5])
        self.production = int(raw[6])

def intersects_circle(x1, y1, x2, y2, cx, cy, r):
    vx = x2 - x1
    vy = y2 - y1
    wx = cx - x1
    wy = cy - y1
    
    segment_len_sq = vx*vx + vy*vy
    if segment_len_sq == 0:
        return math.sqrt(wx*wx + wy*wy) < r
        
    t = (wx*vx + wy*vy) / segment_len_sq
    t = max(0.0, min(1.0, t))
    
    closest_x = x1 + t * vx
    closest_y = y1 + t * vy
    
    dist_sq = (cx - closest_x)**2 + (cy - closest_y)**2
    return dist_sq < r*r

def is_path_blocked(x1, y1, x2, y2, planets, source_id, target_id):
    sun_x, sun_y = 50.0, 50.0
    sun_r = 10.0 + 1.2
    
    if intersects_circle(x1, y1, x2, y2, sun_x, sun_y, sun_r):
        return True
        
    for p in planets:
        if p.id == source_id or p.id == target_id:
            continue
        planet_r = p.radius + 0.8
        if intersects_circle(x1, y1, x2, y2, p.x, p.y, planet_r):
            return True
            
    return False

def get_incoming_threats(my_player, planets, fleets):
    incoming = {p.id: 0 for p in planets}
    for fleet in fleets:
        f_owner = int(fleet[1])
        if f_owner == my_player:
            continue
        f_x = float(fleet[2])
        f_y = float(fleet[3])
        f_angle = float(fleet[4])
        f_ships = int(fleet[6])
        
        dx = math.cos(f_angle)
        dy = math.sin(f_angle)
        
        first_collision_planet = None
        min_collision_t = float('inf')
        
        for p in planets:
            wx = f_x - p.x
            wy = f_y - p.y
            b = 2.0 * (wx*dx + wy*dy)
            c = wx*wx + wy*wy - p.radius*p.radius
            
            disc = b*b - 4.0*c
            if disc >= 0:
                sqrt_disc = math.sqrt(disc)
                t1 = (-b - sqrt_disc) / 2.0
                t2 = (-b + sqrt_disc) / 2.0
                
                t = None
                if t1 > 0:
                    t = t1
                elif t2 > 0:
                    t = t2
                    
                if t is not None and t < min_collision_t:
                    min_collision_t = t
                    first_collision_planet = p
                    
        if first_collision_planet is not None:
            incoming[first_collision_planet.id] += f_ships
            
    return incoming

def agent(obs, config=None):
    if hasattr(obs, "player"):
        my_player = obs.player
        raw_planets = obs.planets
        raw_fleets = obs.fleets if hasattr(obs, "fleets") else []
    else:
        my_player = obs.get("player", 0)
        raw_planets = obs.get("planets", [])
        raw_fleets = obs.get("fleets", [])
        
    planets = [Planet(p) for p in raw_planets]
    incoming_threats = get_incoming_threats(my_player, planets, raw_fleets)
    
    moves = []
    for source in planets:
        if source.owner != my_player:
            continue
            
        threat = incoming_threats.get(source.id, 0)
        net_ships = source.ships - threat
        
        if net_ships <= 15:
            continue
            
        spare_ships = int(net_ships * 0.7)
        
        best_target = None
        best_score = -1.0
        
        for target in planets:
            if target.id == source.id:
                continue
                
            if is_path_blocked(source.x, source.y, target.x, target.y, planets, source.id, target.id):
                continue
                
            dist = math.sqrt((target.x - source.x)**2 + (target.y - source.y)**2)
            
            if target.owner == my_player:
                # Reinforcement
                target_threat = incoming_threats.get(target.id, 0)
                needed = target_threat - target.ships
                if needed > 0:
                    send_amount = min(spare_ships, needed + 2)
                    if send_amount >= 1:
                        score = 10.0 / (dist + 5.0)
                        if score > best_score:
                            best_score = score
                            best_target = (target, send_amount)
            else:
                # Attack
                if target.owner == -1:
                    score = (target.production * 25.0) / ((target.ships + 1.0) * (dist + 8.0))
                else:
                    score = (target.production * 15.0) / ((target.ships + 1.0) * (dist + 10.0))
                    
                if score > best_score:
                    best_score = score
                    best_target = (target, spare_ships)
                    
        if best_target is not None:
            target, send_ships = best_target
            dx = target.x - source.x
            dy = target.y - source.y
            angle = math.atan2(dy, dx)
            moves.append([source.id, angle, send_ships])
            
    return moves
