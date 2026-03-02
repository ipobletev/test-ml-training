import sys
import pygame
import torch
import os
import datetime
import time
from environment import Environment
from model import Agent

def save_outputs(episode_history, model_info, track_name, base_model=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(".output", f"{timestamp}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_file = os.path.join(output_dir, "training.log")
    
    with open(log_file, "w") as f:
        f.write(f"=== TRAINING SESSION: {track_name} ===\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Base Model: {base_model if base_model else 'None (From scratch)'}\n")
        total_dur = model_info.get("total_duration", 0)
        f.write(f"Total Training Duration: {total_dur//60:.0f}m {total_dur%60:.0f}s\n")
        for key, value in model_info.items():
            if key not in ["architecture", "total_duration"]:
                f.write(f"{key}: {value}\n")
        f.write("\n" + model_info["architecture"] + "\n")
        f.write("==============================\n\n")
        
        f.write("Episode,Score,Epsilon,Duration(s)\n")
        for ep, score, eps, dur in episode_history:
            f.write(f"{ep},{score:.2f},{eps:.2f},{dur:.1f}\n")
    
    print(f"📈 Logs saved at: {log_file}")
    return output_dir

def run_training_session(env, config):
    track_name = config["track_name"]
    model_path = config["model_path"]
    training_enabled = config["training_enabled"]
    stuck_detection = config["stuck_detection"]
    num_cars = config.get("num_cars", 1)
    
    state_dim = 6
    action_dim = 5
    agent = Agent(state_dim, action_dim)
    
    output_enabled = True
    
    if model_path:
        print(f"📂 Loading weights from: {model_path}")
        try:
            agent.load(model_path)
            agent.epsilon = 0.2 if training_enabled else 0.0
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    elif not training_enabled:
        print("⚠️ Warning: Simulation without loaded model. Movement will be random.")
        agent.epsilon = 1.0

    batch_size = 64
    episodes = 2000
    episode_history = []
    
    mode_desc = "ACTIVE LEARNING" if training_enabled else "PURE INFERENCE"
    stuck_desc = "Stagnation Detection: ON" if stuck_detection else "Stagnation Detection: OFF"
    print(f"🚀 Starting ({mode_desc} | {stuck_desc}) with {num_cars} cars on: {track_name}")
    
    start_session_time = time.time()
    try:
        for e in range(1, episodes + 1):
            states = env.reset(num_cars=num_cars)
            total_reward = 0
            done = False
            start_episode_time = time.time()
            
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit
                    
                    if env.is_finish_clicked(event):
                        print("🛑 Termination requested by user.")
                        raise KeyboardInterrupt

                    if env.is_out_toggle_clicked(event):
                        output_enabled = not output_enabled
                        status = "ENABLED" if output_enabled else "DISABLED"
                        print(f"💾 Output Generation: {status}")

                actions = [agent.act(s) for s in states]
                next_states, rewards, done = env.step(actions, stuck_detection=stuck_detection)
                
                if training_enabled:
                    for i in range(num_cars):
                        agent.remember(states[i], actions[i], rewards[i], next_states[i], done if isinstance(done, bool) else done[i])
                    
                    for _ in range(min(num_cars, 5)): 
                        agent.replay(batch_size)
                
                states = next_states
                total_reward += sum(rewards) / num_cars
                
                current_session_duration = time.time() - start_session_time
                env.render(e, total_reward, agent.epsilon, output_enabled, training_enabled, session_duration=current_session_duration)
                
                if done:
                    ep_duration = time.time() - start_episode_time
                    if training_enabled:
                        agent.update_target_info()
                        episode_history.append((e, total_reward, agent.epsilon, ep_duration))
                    print(f"✅ Episode {e} finished ({ep_duration:.1f}s). Avg Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    except KeyboardInterrupt:
        print("\n⚠️ Session closed by user. Returning to menu...")
    
    if output_enabled and episode_history:
        total_duration = time.time() - start_session_time
        params = agent.get_parameters()
        params["total_duration"] = total_duration
        output_dir = save_outputs(episode_history, params, track_name, base_model=model_path)
        model_name = os.path.join(output_dir, "car_model_final.pth")
        agent.save(model_name)
        print(f"💾 Progress saved at: {output_dir}")
    else:
        reason = "Output Generation: OFF" if not output_enabled else "No episodes recorded"
        print(f"ℹ️ {reason}. Exiting without saving.")

def main():
    pygame.init()
    env = Environment(render_mode=True)
    
    while True:
        try:
            config = env.show_start_menu()
            if config is None:
                break
            
            run_training_session(env, config)
            
        except SystemExit:
            break
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            break

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
