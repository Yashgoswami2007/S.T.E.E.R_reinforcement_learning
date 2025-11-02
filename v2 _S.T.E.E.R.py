'''YEAH THIS ONE WORKS GREAT'''

import bpy
import random
import math
import numpy as np
import csv
import os
from datetime import datetime

EXPERIENCE_BUFFER_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
LAMBDA = 0.95
PPO_CLIP = 0.2
LEARNING_RATE = 0.0003
PPO_EPOCHS = 4

DRIVE_VALUE = 15.0
STEERING_VALUE = 1.0
MAX_STEPS = 500
GOAL_RADIUS = 3.0

class PPOCarAgent:
    def __init__(self, state_size=8, action_size=5):
        self.state_size = state_size
        self.action_size = action_size
        self.experience_buffer = []
        self.episode_data = []
        
        self.policy_weights = np.random.randn(state_size, action_size) * 0.1
        self.value_weights = np.random.randn(state_size, 1) * 0.1
        self.old_policy_weights = self.policy_weights.copy()
        
    def get_action(self, state):
        logits = np.dot(state, self.policy_weights)
        probs = self.softmax(logits)
        action = np.random.choice(self.action_size, p=probs)
        return action, probs[action]
    
    def get_value(self, state):
        return np.dot(state, self.value_weights)[0]
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def store_experience(self, state, action, reward, next_state, done, prob, goal_reached=False):
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'prob': prob,
            'goal_reached': goal_reached
        }
        self.experience_buffer.append(experience)
        
        if len(self.experience_buffer) > EXPERIENCE_BUFFER_SIZE:
            self.experience_buffer.pop(0)
    
    def save_to_csv(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"car_ppo_goal_training_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['episode', 'step', 'state', 'action', 'reward', 'next_state', 'done', 'prob', 'goal_reached'])
            
            for i, exp in enumerate(self.experience_buffer):
                writer.writerow([
                    i, i,
                    ','.join(map(str, exp['state'])),
                    exp['action'],
                    exp['reward'],
                    ','.join(map(str, exp['next_state'])),
                    exp['done'],
                    exp['prob'],
                    exp['goal_reached']
                ])
        
        print(f"Experience saved to {filename}")
    
    def compute_advantages(self, rewards, values, next_value, dones):
        advantages = np.zeros(len(rewards))
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + GAMMA * LAMBDA * last_advantage * (1 - dones[t])
            last_advantage = advantages[t]
        
        return advantages
    
    def update_policy(self):
        if len(self.experience_buffer) < BATCH_SIZE:
            return
        
        states = np.array([exp['state'] for exp in self.experience_buffer])
        actions = np.array([exp['action'] for exp in self.experience_buffer])
        rewards = np.array([exp['reward'] for exp in self.experience_buffer])
        next_states = np.array([exp['next_state'] for exp in self.experience_buffer])
        dones = np.array([exp['done'] for exp in self.experience_buffer])
        old_probs = np.array([exp['prob'] for exp in self.experience_buffer])
        
        values = np.array([self.get_value(state) for state in states])
        next_value = self.get_value(next_states[-1]) if not dones[-1] else 0
        advantages = self.compute_advantages(rewards, values, next_value, dones)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(PPO_EPOCHS):
            indices = np.random.choice(len(self.experience_buffer), BATCH_SIZE)
            
            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_advantages = advantages[indices]
            batch_old_probs = old_probs[indices]
            
            for i, (state, action, advantage, old_prob) in enumerate(zip(batch_states, batch_actions, batch_advantages, batch_old_probs)):
                logits = np.dot(state, self.policy_weights)
                new_probs = self.softmax(logits)
                new_prob = new_probs[action]
                
                ratio = new_prob / (old_prob + 1e-8)
                
                surr1 = ratio * advantage
                surr2 = np.clip(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantage
                policy_loss = -np.minimum(surr1, surr2)
                
                policy_grad = policy_loss * state.reshape(-1, 1)
                action_mask = np.zeros(self.action_size)
                action_mask[action] = 1
                self.policy_weights -= LEARNING_RATE * policy_grad * action_mask
                
                target = advantage + self.get_value(state)
                value_loss = (target - self.get_value(state)) ** 2
                self.value_weights -= LEARNING_RATE * 0.5 * value_loss * state.reshape(-1, 1)
        
        print("PPO Policy Updated!")


class CarEnvironment:
    def __init__(self):
        self.agent = PPOCarAgent()
        self.current_episode = 0
        self.current_step = 0
        self.current_state = None
        self.total_reward = 0
        self.training_active = False
        self.goal_position = None
        self.car_start_position = None
        self.setup_goal()
        
    def setup_goal(self):
        self.goal_position = np.array([10.0, 0.0, 10.0])
        self.car_start_position = self.get_car_position()
        print(f"Goal set at position: {self.goal_position}")
        print(f"Car start position: {self.car_start_position}")
    
    def get_car_position(self):
        try:
            rig = bpy.context.scene.sna_rbc_rig_collection[0]
            if hasattr(rig, 'location'):
                return np.array([rig.location.x, rig.location.y, rig.location.z])
            else:
                for obj in bpy.context.scene.objects:
                    if obj.type == 'MESH' and "car" in obj.name.lower():
                        return np.array([obj.location.x, obj.location.y, obj.location.z])
        except:
            pass
        return np.array([0.0, 0.0, 0.0])
    
    def get_state(self):
        try:
            rig = bpy.context.scene.sna_rbc_rig_collection[0]
            car_pos = self.get_car_position()
            
            distance_to_goal = np.linalg.norm(car_pos - self.goal_position)
            
            direction_to_goal = (self.goal_position - car_pos) / (distance_to_goal + 1e-8)
            state = [
                rig.rig_drivers.drive / DRIVE_VALUE,
                rig.rig_drivers.steering / STEERING_VALUE,
                car_pos[0] / 50.0,
                car_pos[1] / 50.0,
                car_pos[2] / 50.0,
                direction_to_goal[0],
                direction_to_goal[1],
                direction_to_goal[2]
            ]
            
            return np.array(state)
        except Exception as e:
            print(f"State error: {e}")
            return np.zeros(8)
    
    def calculate_reward(self, state, action, next_state):
        reward = 0.0
        
        car_pos = self.get_car_position()
        current_distance = np.linalg.norm(car_pos - self.goal_position)
        
        if action == 0:
            reward += 0.1
        
        if current_distance <= GOAL_RADIUS:
            reward += 100.0
            print("GOAL REACHED!")
        
        if state[0] > 0:
            reward += 0.05
        
        reward -= 0.01
        
        return reward
    
    def map_action_to_control(self, action):
        if action == 0:
            return DRIVE_VALUE * 0.8, 0.0
        elif action == 1:
            return DRIVE_VALUE * 0.7, STEERING_VALUE * 0.3
        elif action == 2:
            return DRIVE_VALUE * 0.7, -STEERING_VALUE * 0.3
        elif action == 3:
            return DRIVE_VALUE * 0.5, STEERING_VALUE * 0.7
        elif action == 4:
            return DRIVE_VALUE * 0.5, -STEERING_VALUE * 0.7
        else:
            return 0.0, 0.0
    
    def reset_episode(self):
        self.current_step = 0
        self.total_reward = 0
        self.current_state = self.get_state()
        self.current_episode += 1
        
        try:
            rig = bpy.context.scene.sna_rbc_rig_collection[0]
            rig.rig_drivers.drive = 0.0
            rig.rig_drivers.steering = 0.0
        except:
            pass
        
        print(f"Starting Episode {self.current_episode}")
    
    def is_goal_reached(self):
        car_pos = self.get_car_position()
        distance_to_goal = np.linalg.norm(car_pos - self.goal_position)
        return distance_to_goal <= GOAL_RADIUS
    
    def step(self):
        if not self.training_active:
            return
        
        action, action_prob = self.agent.get_action(self.current_state)
        
        drive, steering = self.map_action_to_control(action)
        try:
            rig = bpy.context.scene.sna_rbc_rig_collection[0]
            rig.rig_drivers.drive = drive
            rig.rig_drivers.steering = steering
            print(f"Action: {action}, Drive: {drive}, Steering: {steering}")
        except Exception as e:
            print(f"Car control error: {e}")
            return
        
        next_state = self.get_state()
        
        goal_reached = self.is_goal_reached()
        
        reward = self.calculate_reward(self.current_state, action, next_state)
        self.total_reward += reward
        
        done = self.current_step >= MAX_STEPS or goal_reached
        
        self.agent.store_experience(
            self.current_state, action, reward, next_state, done, action_prob, goal_reached
        )
        
        self.current_state = next_state
        self.current_step += 1
        
        if self.current_step % 10 == 0 or goal_reached:
            car_pos = self.get_car_position()
            distance = np.linalg.norm(car_pos - self.goal_position)
            print(f"Episode {self.current_episode}, Step {self.current_step}, Distance: {distance:.1f}, Total Reward: {self.total_reward:.2f}")
        
        if done:
            success = "SUCCESS" if goal_reached else "TIME OUT"
            print(f"Episode {self.current_episode} completed! {success} Total reward: {self.total_reward:.2f}")
            
            if len(self.agent.experience_buffer) >= BATCH_SIZE:
                self.agent.update_policy()
            
            if self.current_episode % 5 == 0:
                self.agent.save_to_csv()
            
            self.reset_episode()


car_env = CarEnvironment()

def training_frame_handler(scene):
    car_env.step()

class StartPPOTrainingOperator(bpy.types.Operator):
    bl_idname = "wm.start_ppo_training"
    bl_label = "Start PPO Goal Training"
    bl_description = "Start PPO reinforcement learning for goal-reaching"
    
    def execute(self, context):
        global car_env
        car_env.training_active = True
        car_env.reset_episode()
        
        if training_frame_handler not in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.append(training_frame_handler)
        
        self.report({'INFO'}, "PPO Goal Training Started!")
        print("Starting PPO Goal-Oriented Training!")
        return {'FINISHED'}

class StopPPOTrainingOperator(bpy.types.Operator):
    bl_idname = "wm.stop_ppo_training"
    bl_label = "Stop PPO Training"
    bl_description = "Stop PPO reinforcement learning"
    
    def execute(self, context):
        global car_env
        car_env.training_active = False
        
        car_env.agent.save_to_csv()
        
        if training_frame_handler in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.remove(training_frame_handler)
            
        self.report({'INFO'}, "PPO Training Stopped!")
        print("PPO Training Stopped!")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(StartPPOTrainingOperator)
    bpy.utils.register_class(StopPPOTrainingOperator)
    
    print("PPO Goal-Oriented Car Training System Ready!")
    print("Run 'Start PPO Goal Training' from F3 menu to begin!")

def unregister():
    global car_env
    car_env.training_active = False
    
    if training_frame_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(training_frame_handler)
    
    bpy.utils.unregister_class(StartPPOTrainingOperator)
    bpy.utils.unregister_class(StopPPOTrainingOperator)

register()
bpy.ops.wm.start_ppo_training()
