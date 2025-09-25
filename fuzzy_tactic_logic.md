# FuzzyTactic Logic Overview

## 1. Utilities

```python
    triag(x,a,b,c): ...    # Fuzzy triangle membership (close, far, fast, slow)
    wrap180(d): ...        # Normalize angle to [-180, 180]
    get_heading_degrees(ship_state): ...
    intercept_point(ship_pos, ship_vel, bullet_speed, target_pos, target_vel): ...
    calculate_threat_priority(asteroid, ship_pos, ship_vel): ...
    find_closest_threat(asteroids, ship_pos): ...
```

- **triag** -> defines fuzzy sets for distance/velocity.  
- **wrap180** -> keeps heading errors manageable.  
- **get_heading_degrees** -> finds the ship’s facing direction.  
- **intercept_point** -> predictive aiming (where bullet & asteroid meet).  
- **calculate_threat_priority** -> ranks asteroids (distance, closing speed, size).  
- **find_closest_threat** -> nearest asteroid by distance.  

---

## 2. Danger Assessment

```python
very_close = triag(dist, 0,80,160)
close      = triag(dist, 120,200,300)
medium     = triag(dist, 250,400,600)
far        = triag(dist, 500,700,1000)

fast_approach = triag(speed, 50,150,300)
slow_approach = triag(speed, 10,50,100)
moving_away   = triag(speed,-200,-50,10)

danger_level = max(very_close, min(close, max(fast_approach, slow_approach)))
```

- Combines **distance** and **approach speed** into a single danger score.  
- This danger level drives which behavior (mode) is selected.  

---

## 3. Modes (Decision Logic)

The controller switches between four modes depending on danger:

### (a) **Critical Dodge (panic mode)**  

```python
if dist < 120 and speed > 30:
    # Pick sideways dodge with fewer asteroids
    perp = perp1 if score1 > score2 else perp2
    thrust, turn_rate = 150.0, dodge_err * 4.0
```

- **Trigger:** very close asteroid rushing in.  
- **Action:** emergency sidestep -> strong sideways thrust, sharp turn.  

---

### (b) **Danger Drift (moonwalking away)**  

```python
elif danger_level > 0.3:
    thrust = -120.0
    turn_rate = aim_err * 3.0
```

- **Trigger:** moderately high danger.  
- **Action:** back away with reverse thrust, keep asteroid in view.  

---

### (c) **Engagement (attack mode)**  

```python
elif medium > 0.2:
    thrust = 80.0
    best = max(asteroids, key=lambda a: calculate_threat_priority(...))
    ix, iy = intercept_point(...)
    heading_err = wrap180(desired_heading - heading)
    turn_rate = heading_err * 3.0
```

- **Trigger:** asteroid at medium distance, not too dangerous.  
- **Action:** thrust forward slowly, aim using intercept prediction, fire if aligned.  

---

### (d) **Far Approach (cruising)**  

```python
else:
    thrust = 120.0
    turn_rate = approach_err * 2.0
```

- **Trigger:** asteroid far away or safe.  
- **Action:** cruise toward target asteroid with gentle turns.  

---

## 4. Weapons

```python
if dist > 100:
    fire = abs(heading_err) < 20 and target_distance < 700
else:
    fire = False

drop_mine = (dist < 60 and asteroid_size >= 3 and speed > 80)
```

- **Fire gun** -> when within 20° aim and <700 units.  
- **Drop mine** -> if asteroid is very close, large, and closing fast.  

---

## 5. Safety Clamp

```python
if hasattr(ship_state,"thrust_range"):
    thrust = max(lo, min(hi, thrust))
if hasattr(ship_state,"turn_rate_range"):
    turn_rate = max(lo, min(hi, turn_rate))
```

- Ensures thrust and turn rate stay within ship’s limits.  

---

## 6. Flow of Thought

1. **Look around** -> detect asteroids.  
2. **Pick closest** -> compute its distance & speed.  
3. **Fuzzify danger** -> translate numbers into categories (close, medium, far; fast, slow, away).  
4. **Decide behavior** -> Panic -> Drift -> Engage -> Cruise.  
5. **Act** -> thrust, turn, fire, drop mine.  
6. **Clamp** -> keep actions within safe ship bounds.  

---

**Summary:**  

- Too close & fast -> **panic dodge**.  
- Danger but not panic -> **back off**.  
- Medium safe range -> **engage & shoot**.  
- Far -> **cruise in**.  
- Weapons fire only when aligned, mines drop under extreme close danger.  

This is essentially a **state machine** driven by fuzzy danger assessment.  

