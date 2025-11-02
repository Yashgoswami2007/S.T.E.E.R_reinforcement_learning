'''simulated training environment for enhanced learning  S,T,E,E,R'''
import bpy
import random
import math

# --- Settings ---
DRIVE_VALUE = 20.0
STEERING_VALUE = 1.0
RANDOM_MOVE_INTERVAL = 30  # frames between random movement decisions
MIN_RANDOM_MOVE = 20       # minimum frames for a random move
MAX_RANDOM_MOVE = 60       # maximum frames for a random move
EXPLORATION_TIME = 10000    # total frames for exploration session

# Exploration state
exploration_active = True
current_action = None
action_duration = 0
action_counter = 0
total_frames = 0

def get_exploration_action():
    actions = [
        # Forward movements
        {"drive": DRIVE_VALUE, "steering": 0.0, "weight": 3},           
        {"drive": DRIVE_VALUE * 0.8, "steering": STEERING_VALUE * 0.7, "weight": 2},  
        {"drive": DRIVE_VALUE * 0.8, "steering": -STEERING_VALUE * 0.7, "weight": 2}, 
        {"drive": DRIVE_VALUE * 0.6, "steering": STEERING_VALUE, "weight": 1},        
        {"drive": DRIVE_VALUE * 0.6, "steering": -STEERING_VALUE, "weight": 1},       
        
        # Reverse and turn
        {"drive": -DRIVE_VALUE * 0.4, "steering": STEERING_VALUE * 0.5, "weight": 1}, 
        {"drive": -DRIVE_VALUE * 0.4, "steering": -STEERING_VALUE * 0.5, "weight": 1},
        
        # Turning in place
        {"drive": 0.0, "steering": STEERING_VALUE * 1.2, "weight": 1}, 
        {"drive": 0.0, "steering": -STEERING_VALUE * 1.2, "weight": 1}, 
        
        # Pause and observe
        {"drive": 0.0, "steering": 0.0, "weight": 1},                   
    ]
    
    
    weighted_actions = []
    for action in actions:
        weighted_actions.extend([action] * action["weight"])
    
    return random.choice(weighted_actions)

def update_exploration_movement():
    global current_action, action_duration, action_counter, total_frames, exploration_active
    
    try:
        rig = bpy.context.scene.sna_rbc_rig_collection[0]
        
        total_frames += 1
        if total_frames >= EXPLORATION_TIME:
            exploration_active = False
            rig.rig_drivers.drive = 0.0
            rig.rig_drivers.steering = 0.0
            print("🚗 Exploration session completed!")
            return
        
        # Get new random action if needed
        if current_action is None or action_counter >= action_duration:
            current_action = get_exploration_action()
            action_duration = random.randint(MIN_RANDOM_MOVE, MAX_RANDOM_MOVE)
            action_counter = 0
            
            # Print current action for debugging
            drive_dir = "forward" if current_action["drive"] > 0 else "backward" if current_action["drive"] < 0 else "stopped"
            steer_dir = "right" if current_action["steering"] > 0 else "left" if current_action["steering"] < 0 else "straight"
            print(f"🎯 Car action: {drive_dir}, turning {steer_dir} for {action_duration} frames")
        
        
        rig.rig_drivers.drive = current_action["drive"]
        rig.rig_drivers.steering = current_action["steering"]
        
        action_counter += 1
        
    except Exception as e:
        print(f"🚗 Exploration error: {e}")
        exploration_active = False

def frame_handler(scene):
    if exploration_active:
        update_exploration_movement()

def start_exploration():
    global exploration_active, current_action, action_duration, action_counter, total_frames
    
    exploration_active = True
    current_action = None
    action_duration = 0
    action_counter = 0
    total_frames = 0
    
    
    if frame_handler not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(frame_handler)
    

class StartExplorationOperator(bpy.types.Operator):
    bl_idname = "wm.start_exploration"
    bl_label = "Start Car Exploration"
    bl_description = "Start autonomous car exploration with random movements"
    
    def execute(self, context):
        start_exploration()
        self.report({'INFO'}, "Car exploration started!")
        return {'FINISHED'}

class StopExplorationOperator(bpy.types.Operator):
    bl_idname = "wm.stop_exploration"
    bl_label = "Stop Car Exploration"
    bl_description = "Stop autonomous car exploration"
    
    def execute(self, context):
        global exploration_active
        
        exploration_active = False
        try:
            rig = bpy.context.scene.sna_rbc_rig_collection[0]
            rig.rig_drivers.drive = 0.0
            rig.rig_drivers.steering = 0.0
        except:
            pass
        
        # Remove handler
        if frame_handler in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.remove(frame_handler)
            
        self.report({'INFO'}, "Car exploration stopped!")
        print("🛑 Car exploration stopped by user")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(StartExplorationOperator)
    bpy.utils.register_class(StopExplorationOperator)
    
    # Auto-start exploration
    start_exploration()

def unregister():
    global exploration_active
    
    exploration_active = False
    if frame_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(frame_handler)
    
    bpy.utils.unregister_class(StartExplorationOperator)
    bpy.utils.unregister_class(StopExplorationOperator)

# Start the exploration
register()
