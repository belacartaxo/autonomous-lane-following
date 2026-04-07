import gymnasium as gym
from gymnasium import spaces
import numpy as np
from controller import Supervisor

class WebotsVehicleEnv(gym.Env):
    def __init__(self):
        super(WebotsVehicleEnv, self).__init__()
        
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Dispositivos
        self.lidar = self.robot.getDevice("Sick LMS 291")
        self.lidar.enable(self.timestep)
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)

        # Referência do Robô e Posição Inicial Automática
        self.vehicle_node = self.robot.getSelf()
        self.translation_field = self.vehicle_node.getField("translation")
        self.rotation_field = self.vehicle_node.getField("rotation")
        
        # SALVA A POSIÇÃO QUE VOCÊ DEFINIU NO WEBOTS
        self.initial_translation = self.translation_field.getSFVec3f()
        self.initial_rotation = self.rotation_field.getSFRotation()

        # --- DEFINIÇÃO DOS ESPAÇOS (O que estava faltando!) ---
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32), 
            high=np.array([0.5, 1.0], dtype=np.float32), 
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(low=0, high=10, shape=(72,), dtype=np.float32),
            "camera": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        })
        # -------------------------------------------------------

        # Motores - Usando nomes universais do BMW X5
        self.left_steering = self.robot.getDevice("left_steer")
        self.right_steering = self.robot.getDevice("right_steer")
        
        self.wheels = []
        possible_wheels = ["left_front_wheel", "right_front_wheel", "left_rear_wheel", "right_rear_wheel"]
        for name in possible_wheels:
            wheel = self.robot.getDevice(name)
            if wheel:
                self.wheels.append(wheel)
                wheel.setPosition(float('inf'))
                wheel.setVelocity(0.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # RETORNA AO LUGAR SALVO NO INIT
        self.translation_field.setSFVec3f(self.initial_translation)
        self.rotation_field.setSFRotation(self.initial_rotation)
        
        self.vehicle_node.resetPhysics()
        self.robot.step(self.timestep)
        
        obs = {
            "lidar": np.nan_to_num(np.array(self.lidar.getRangeImage()), posinf=10.0),
            "camera": self.get_camera_image()
        }
        return obs, {}

    def get_camera_image(self):
        # Pega a imagem crua do Webots
        image_data = self.camera.getImage()
        # Converte para array numpy (Altura, Largura, Canais)
        img = np.frombuffer(image_data, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
        # Converte de BGRA para RGB (remove o canal Alpha e inverte B/R)
        img_rgb = img[:, :, [2, 1, 0]]
        return img_rgb


    def step(self, action):
        steer_angle = float(action[0])
        # action[1] varia de -1 a 1. Se for negativo, o carro dá ré.
        velocity = float(action[1]) * 20.0 

        # Aplica direção
        if self.left_steering and self.right_steering:
            self.left_steering.setPosition(steer_angle)
            self.right_steering.setPosition(steer_angle)

        # Aplica velocidade apenas nas rodas encontradas
        for wheel in self.wheels:
            wheel.setVelocity(velocity)

        self.robot.step(self.timestep)

        # (O restante do seu código de capturar observações e reward continua igual...)
        lidar_values = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        obs = {
            "lidar": np.nan_to_num(lidar_values, posinf=10.0),
            "camera": self.get_camera_image()
        }
        
        # Critério de parada (Done) se bater
        done = False
        if np.min(lidar_values) < 0.5:
            done = True
            
        return obs, 0.0, done, False, {}

# Bloco para testar se a classe funciona
if __name__ == "__main__":
    env = WebotsVehicleEnv()
    print("Ambiente Gymnasium criado com sucesso!")
    obs, _ = env.reset()
    print(f"Formato do LiDAR: {obs['lidar'].shape}") # Deve imprimir (72,)
    print(f"Formato da Câmera: {obs['camera'].shape}") # Deve imprimir (64, 64, 3)