import sys
import pygame
import torch
import os
import datetime
from environment import Environment
from model import Agent

def save_outputs(episode_history, model_info, track_name, base_model=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"{timestamp}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_file = os.path.join(output_dir, "training.log")
    
    with open(log_file, "w") as f:
        f.write(f"=== ENTRENAMIENTO: {track_name} ===\n")
        f.write(f"Fecha: {datetime.datetime.now()}\n")
        f.write(f"Modelo Base: {base_model if base_model else 'Ninguno (Desde cero)'}\n")
        for key, value in model_info.items():
            if key != "architecture":
                f.write(f"{key}: {value}\n")
        f.write("\n" + model_info["architecture"] + "\n")
        f.write("==============================\n\n")
        
        f.write("Episode,Score,Epsilon\n")
        for ep, score, eps in episode_history:
            f.write(f"{ep},{score:.2f},{eps:.2f}\n")
    
    print(f"📈 Logs guardados en: {log_file}")
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
        print(f"📂 Cargando pesos desde: {model_path}")
        try:
            agent.load(model_path)
            agent.epsilon = 0.2 if training_enabled else 0.0
        except Exception as e:
            print(f"❌ Error al cargar modelo: {e}")
    elif not training_enabled:
        print("⚠️ Advertencia: Simulación sin modelo cargado. El movimiento será aleatorio.")
        agent.epsilon = 1.0

    batch_size = 64
    episodes = 2000
    episode_history = []
    
    mode_desc = "APRENDIZAJE ACTIVO" if training_enabled else "INFERENCIA PURA"
    stuck_desc = "Detección Estancamiento: ON" if stuck_detection else "Detección Estancamiento: OFF"
    print(f"🚀 Iniciando ({mode_desc} | {stuck_desc}) con {num_cars} autos en: {track_name}")
    
    try:
        for e in range(1, episodes + 1):
            states = env.reset(num_cars=num_cars)
            total_reward = 0
            done = False
            
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit
                    
                    if env.is_finish_clicked(event):
                        print("🛑 Finalización solicitada por el usuario.")
                        raise KeyboardInterrupt

                    if env.is_out_toggle_clicked(event):
                        output_enabled = not output_enabled
                        status = "ACTIVADA" if output_enabled else "DESACTIVADA"
                        print(f"💾 Generación de Output: {status}")

                actions = [agent.act(s) for s in states]
                next_states, rewards, done = env.step(actions, stuck_detection=stuck_detection)
                
                if training_enabled:
                    for i in range(num_cars):
                        # Only remember if the car was alive during this step (handled by environment giving 0 reward/True done if already dead)
                        # However, for efficiency, we can filter or just push all. 
                        # DQN replay will handle it.
                        agent.remember(states[i], actions[i], rewards[i], next_states[i], done if isinstance(done, bool) else done[i])
                    
                    # Train more if we have more cars providing data
                    for _ in range(min(num_cars, 5)): 
                        agent.replay(batch_size)
                
                states = next_states
                total_reward += sum(rewards) / num_cars
                
                env.render(e, total_reward, agent.epsilon, output_enabled, training_enabled)
                
                if done:
                    if training_enabled:
                        agent.update_target_info()
                        episode_history.append((e, total_reward, agent.epsilon))
                    print(f"✅ Episodio {e} finalizado. Avg Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    except KeyboardInterrupt:
        print("\n⚠️ Sesión cerrada por el usuario. Volviendo al menú...")
    
    if output_enabled and episode_history:
        output_dir = save_outputs(episode_history, agent.get_parameters(), track_name, base_model=model_path)
        model_name = os.path.join(output_dir, "car_model_final.pth")
        agent.save(model_name)
        print(f"💾 Avances guardados en: {output_dir}")
    else:
        reason = "Modo Generar Output: OFF" if not output_enabled else "Sin episodios registrados"
        print(f"ℹ️ {reason}. Saliendo sin guardar.")

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
            print(f"💥 Error inesperado: {e}")
            break

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
