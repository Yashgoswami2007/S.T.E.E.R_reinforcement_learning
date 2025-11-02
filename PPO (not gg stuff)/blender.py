import bpy
import bgl
import json
import socket
import threading
import time
import mathutils

HOST = "172.16.5.93"
PORT = 5001
CAR_NAME = "Car"
GOAL_NAME = "Goal"
DT = 1.0 / 30.0  # step frequency

server_socket = None
client_sock = None
client_addr = None
running = True

def get_car_state():
    car = bpy.data.objects.get(CAR_NAME)
    goal = bpy.data.objects.get(GOAL_NAME)
    if car is None or goal is None:
        return None
    pos = car.location
    rot = car.rotation_euler
    vel = mathutils.Vector((0.0, 0.0, 0.0))
    if car.rigid_body:
        vel = car.rigid_body.linear_velocity
    dist_to_goal = (goal.location - pos).length
    
    state = [pos.x, pos.y, pos.z, rot.z, vel.x, vel.y, dist_to_goal]
    return state

def apply_action(action):
    # action = [steer, throttle, brake]
    car = bpy.data.objects.get(CAR_NAME)
    if car is None:
        return
    steer, throttle, brake = action
    forward = car.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 1.0, 0.0))
    if car.rigid_body:
        force = forward * float(throttle) * 200.0  # tune constant
        car.rigid_body.apply_force(force, True)
        
        if brake > 0.0:
            car.rigid_body.apply_force(-vel * float(brake) * 50.0, True)
    
    car.rotation_euler.z += float(steer) * 0.02

def reset_environment():
    
    car = bpy.data.objects.get(CAR_NAME)
    if car and car.rigid_body:
        car.location = mathutils.Vector((0.0, 0.0, 0.25))
        car.rotation_euler = mathutils.Euler((0.0, 0.0, 0.0))
        car.rigid_body.linear_velocity = mathutils.Vector((0,0,0))
        car.rigid_body.angular_velocity = mathutils.Vector((0,0,0))

def client_thread(conn, addr):
    global running, client_sock
    client_sock = conn
    client_addr = addr
    print("Trainer connected:", addr)
    try:
        last_time = time.time()
        while running:
            state = get_car_state()
            if state is None:
                time.sleep(1.0)
                continue
            
            msg = {"type":"state", "state": state, "ts": time.time()}
            conn.sendall((json.dumps(msg) + "\n").encode("utf-8"))
            
            data = b""
            # read a line
            while not data.endswith(b"\n"):
                chunk = conn.recv(4096)
                if not chunk:
                    running = False
                    break
                data += chunk
            if not running:
                break
            try:
                j = json.loads(data.decode("utf-8").strip())
            except Exception as e:
                print("JSON parse err", e)
                continue
            if j.get("type") == "action":
                apply_action(j.get("action", [0,0,0]))
            elif j.get("type") == "reset":
                reset_environment()
            
            bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
            
            time.sleep(DT)
    except Exception as e:
        print("client thread error:", e)
    finally:
        conn.close()
        print("Trainer disconnected")

def server_loop():
    global server_socket, running
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print("Blender bridge listening on", HOST, PORT)
    while running:
        conn, addr = server_socket.accept()
        t = threading.Thread(target=client_thread, args=(conn, addr), daemon=True)
        t.start()

def stop_server():
    global running, server_socket
    running = False
    try:
        server_socket.close()
    except:
        pass

t = threading.Thread(target=server_loop, daemon=True)
t.start()
print("Blender bridge started")