import sys
import os
import ctypes
import time

WEBOTS_HOME = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")

webots_python_path = os.path.join(WEBOTS_HOME, "lib", "controller", "python")
webots_controller_dll = os.path.join(WEBOTS_HOME, "lib", "controller", "Controller.dll")

if webots_python_path not in sys.path:
    sys.path.append(webots_python_path)

ctypes.CDLL(webots_controller_dll)

from env.webots_critical_env import WebotsCriticalVehicleEnv


def main():
    env = WebotsCriticalVehicleEnv()
    obs, info = env.reset()

    print("Ambiente crítico carregado com sucesso.")
    print("A testar trigger do pedestre...")

    for step in range(2000):
        action = [0.0, 0.3]

        obs, reward, done, truncated, info = env.step(action)

        vehicle_position = env.translation_field.getSFVec3f()

        print(
            f"Step: {step:04d} | "
            f"Posição do veículo: {vehicle_position} | "
            f"Reward: {reward:.2f} | "
            f"Done: {done}"
        )

        time.sleep(0.01)

        if done:
            print("Episódio terminou.")
            break

    print("Teste finalizado.")


if __name__ == "__main__":
    main()