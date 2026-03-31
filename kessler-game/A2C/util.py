#Author: Kyle Nguyen
#Date: September 2025
#Description: Utility functions for fuzzy logic controller and other controllers

import math

# triangular membership function
def triag(x, a, b, c):# slope magic
    if b ==a: return 0.0
    if c ==b: return 0.0
    if x <= a or x >= c:# outside the triangle
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a)  #on the upslope, linear interpolation from a to b
    if b < x < c:
        return (c - x) / (c - b)  #on the downslope, linear interpolation from b to c
    
    
def trap(x, a, b, c, d):# trapezoidal membership function
    if x <= a or x >= d:# outside the trapezoid
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)  #on the upslope, linear interpolation from a to b
    if b <= x <= c:
        return 1.0  #top of the trapezoid
    if c < x < d:
        return (d - x) / (d - c)  #on the downslope, linear interpolation from c to d

# makes the numbers wrap around -180 to +180
def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


# toroidal geometry helpers, map wraps like pac man
# these used to be in hybrid_fuzzy.py but everything needs them

SHIP_RADIUS = 20.0  # from the kessler API docs

def wrap_delta(d, size):
    #shortest signed delta on one wrapping axis
    d = d % size
    if d > size / 2:
        d -= size
    return d

def toro_dx_dy(sx, sy, ax, ay, map_size):
    #shortest dx, dy from ship to asteroid on toroidal map
    w, h = map_size
    return wrap_delta(ax - sx, w), wrap_delta(ay - sy, h)

def toro_dist(sx, sy, ax, ay, map_size):
    dx, dy = toro_dx_dy(sx, sy, ax, ay, map_size)
    return math.hypot(dx, dy)


# shared threat finding, one version used by all controllers

def find_closest_threat(asteroids, ship_pos, map_size):
    #find the asteroid with smallest gap (edge to edge) using wrapping
    closest_gap = float('inf')
    closest = None

    for a in asteroids:
        ax, ay = a.position
        center_dist = toro_dist(ship_pos[0], ship_pos[1], ax, ay, map_size)
        gap = center_dist - getattr(a, "radius", 0.0) - SHIP_RADIUS
        if gap < closest_gap:
            closest_gap = gap
            closest = a

    return closest, max(closest_gap, 1.0)


# threat priority, higher = scarier
# now uses toroidal wrapping so it picks the right target near edges
def calculate_threat_priority(asteroid, ship_pos, ship_vel, map_size):
    ax, ay = asteroid.position
    dx, dy = toro_dx_dy(ship_pos[0], ship_pos[1], ax, ay, map_size)
    distance = math.hypot(dx, dy)

    avx, avy = getattr(asteroid, "velocity", (0.0, 0.0))
    closing_speed = ((avx - ship_vel[0]) * dx + (avy - ship_vel[1]) * dy) / max(distance, 1)

    size = getattr(asteroid, "size", 2)

    #closer = higher priority, rushing toward us = higher, smaller = slightly higher
    priority = (1000.0 / distance) + max(closing_speed, 0) / 50.0 + (5 - size)
    return priority


#try to guess where to shoot
#now takes optional map_size for wrapping, None = old behavior
def intercept_point(ship_pos, ship_vel, target_pos, target_vel, map_size=None):
    
    if map_size is not None:
        dx, dy = toro_dx_dy(ship_pos[0], ship_pos[1],
                            target_pos[0], target_pos[1], map_size)
    else:
        dx, dy = target_pos[0] - ship_pos[0], target_pos[1] - ship_pos[1] #vector from ship to target

    dvx, dvy = target_vel[0] - ship_vel[0], target_vel[1] - ship_vel[1]#relative vel, how the target is moving compared to us

    bullet_speed = 800.0
    #Quadratic problem in time t
    #LHS: squared distance to target at time t
    #RHS: squared distance bullet travels in time t
    #Solve for t, then use t to find intercept point
    # a*t^2 + b*t + c = 0
    a = dvx**2 + dvy**2 - bullet_speed**2
    b = 2 * (dx*dvx + dy*dvy)
    c = dx**2 + dy**2

    delta = b*b - 4*a*c
    if delta < 0 or abs(a) < 1e-6:
        #no solution, just aim where it is right now
        return (ship_pos[0] + dx, ship_pos[1] + dy)

    t1 = (-b + math.sqrt(delta)) / (2*a)
    t2 = (-b - math.sqrt(delta)) / (2*a)
    t_candidates = [t for t in (t1, t2) if t > 0]#keeps only positive times

    if not t_candidates:
        return (ship_pos[0] + dx, ship_pos[1] + dy)

    #pick the soonest intercept time
    t = min(t_candidates)

    # predicted position relative to ship
    # toro_dx_dy already handled the wrap for us
    ix = ship_pos[0] + dx + dvx * t
    iy = ship_pos[1] + dy + dvy * t
    return (ix, iy)