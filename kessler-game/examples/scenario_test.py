# kessler-game/examples/scenario_test.py
import time
from git import Actor
from kesslergame import KesslerGame, GraphicsType
from ActorCriticFuzz.Actor_Critic import ActorCriticController 
import scenarios as sc  


#SCENARIO = sc.donut_ring()
#SCENARIO = sc.vertical_wall_left()
SCENARIO = sc.stock_scenario()
#SCENARIO = sc.spiral_arms()
#SCENARIO = sc.sniper_practice()
#SCENARIO = sc.crossing_lanes()
#SCENARIO = sc.asteroid_rain()
#SCENARIO = sc.giants_with_kamikaze()
#SCENARIO = sc.donut_ring_closing()
#SCENARIO = sc.rotating_cross()
#SCENARIO = sc.moving_maze_right()

game_settings = {
    'perf_tracker': True,
    'graphics_type': GraphicsType.Tkinter,
    'realtime_multiplier': 1,
    'graphics_obj': None,
    'frequency': 30
}

game = KesslerGame(settings=game_settings)

pre = time.perf_counter()

score, perf_data = game.run(
    scenario=SCENARIO,
    controllers=[ActorCriticController()]
)

print('Scenario eval time:', time.perf_counter() - pre)
print(score.stop_reason)
print('Asteroids hit:', [team.asteroids_hit for team in score.teams])
print('Deaths:', [team.deaths for team in score.teams])
print('Accuracy:', [team.accuracy for team in score.teams])
print('Mean eval time:', [team.mean_eval_time for team in score.teams])

# ------------------------------------------------------------
# INTRO: BASIC TARGETING
# Teaches the AI how to aim, shoot, and move in simple situations
# No pressure, very readable patterns
# Delete the first 2 or 3 if you want
# ------------------------------------------------------------
# INTRO_TARGETING_GROUP = 
# [
#     sc.single_target_practice(),
#     sc.dual_static_targets(),
#     sc.donut_ring(),
#     sc.slow_crossing_paths(),
#     sc.lane_switcher(),
#     sc.stock_scenario(),
# ]


# ------------------------------------------------------------
# LINEAR MOVEMENT: EASY DODGING
# Teaches the AI how to dodge simple straight moving threats
# Everything moves in clear directions
# ------------------------------------------------------------
# LINEAR_FLOW_GROUP = 
# [
#     sc.staggered_fall(),
#     sc.vertical_wall_left(),
#     sc.asteroid_rain(),
#     sc.horizontal_gate_runner(),
# ]


# ------------------------------------------------------------
# RING: CENTER PRESSURE
# Teaches the AI how to deal with pressure around itself
# Focus on staying alive when surrounded or collapsing inward
# ------------------------------------------------------------
# RING_AND_COLLAPSE_GROUP = 
# [
#     sc.donut_ring_closing(),
#     sc.inner_outer_rings(),
# ]


# ------------------------------------------------------------
# LANES: GRID TRAFFIC
# Teaches the AI how to read patterns and move through traffic
# Multiple directions at once
# ------------------------------------------------------------
# LANE_AND_GRID_GROUP = 
# [
#     sc.crossing_lanes(),
#     sc.phase_shift_grid(),
#     sc.diagonal_grid_fast(),
# ]


# ------------------------------------------------------------
# MAZE: PATH FINDING
# Teaches the AI how to find safe paths and navigate tight spaces
# Focus on positioning instead of just reacting
# ------------------------------------------------------------
# MAZE_AND_PATHING_GROUP = 
# [
#     sc.moving_maze_right(),
#     sc.s_curve_chokepoint(),
# ]


# ------------------------------------------------------------
# CURVED: ROTATING MOTION
# Teaches the AI to predict movement and not just in straight lines
# Important for aiming and avoiding harder patterns
# ------------------------------------------------------------
# CURVE_AND_ORBIT_GROUP = 
# [
#     sc.spiral_arms(),
#     sc.double_orbit_with_darts(),
#     sc.rotating_cross(),
# ]


# ------------------------------------------------------------
# EDGE: WRAP PRESSURE
# Teaches the AI to watch edges and react to side attacks
# Danger comes from off screen or both sides
# ------------------------------------------------------------
# EDGE_AND_WRAP_PRESSURE_GROUP = 
# [
#     sc.wrap_wall_light(),
#     sc.wrap_pincer(),
# ]


# ------------------------------------------------------------
# WAVE: SURROUND ATTACKS
# Teaches the AI to survive multiple attacks at once
# Pressure comes from different angles repeatedly
# ------------------------------------------------------------
# WAVE_AND_SURROUND_GROUP = 
# [
#     sc.corner_wave_pairs(),
#     sc.corner_shockwaves(),
# ]


# ------------------------------------------------------------
# PRIORITY / TARGET CHOICE
# Teaches the AI what to focus on first, big or small threats
# Important for decision making
# ------------------------------------------------------------
# MIXED_PRIORITY_GROUP = 
# [
#     sc.giants_with_kamikaze(),
# ]


# ------------------------------------------------------------
# COMPRESSION: HARD SURVIVAL
# Teaches the AI to survive when space gets very tight High Pressure
# ------------------------------------------------------------
# SPACE_COMPRESSION_GROUP = 
# [
#     sc.pinch_chamber(),
# ]
