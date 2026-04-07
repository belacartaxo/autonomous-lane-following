from main_controller import WebotsVehicleEnv
import numpy as np

env = WebotsVehicleEnv()
obs, _ = env.reset()

print("Iniciando Teste com Agente Aleatório...")

for episode in range(5):
    obs, _ = env.reset()
    done = False
    truncated = False
    step_count = 0
    
    while not (done or truncated):
        # Gera ações aleatórias entre os limites definidos
        action = env.action_space.sample() 
        
        # Aplica no ambiente
        obs, reward, done, truncated, info = env.step(action)
        
        step_count += 1
        if step_count % 10 == 0:
            print(f"Episódio {episode} - Passo {step_count} - LiDAR Min: {np.min(obs['lidar']):.2f}m")

    print(f"Episódio {episode} finalizado após {step_count} passos.")