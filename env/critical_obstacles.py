import numpy as np


class CriticalObstacleManager:
    """
    Manager for critical dynamic obstacle scenarios.

    The pedestrian starts from its current position in the Webots world.
    When the vehicle enters a 2D risk radius around the pedestrian, the pedestrian
    walks forward for a fixed distance.
    """

    def __init__(self, supervisor, vehicle_translation_field):
        self.supervisor = supervisor
        self.vehicle_translation_field = vehicle_translation_field

        self.scenarios = [
            {
                "def_name": "PEDESTRIAN_1",
                "trigger_distance": 25.0,
                "walk_direction": [1.0, 0.0, 0.0],
                "walk_distance": 24.0,
                "speed": 0.08,
                "active": False,
                "completed": False,
            }
        ]

        self.obstacles = []
        self._load_obstacles()

    def _load_obstacles(self):
        for scenario in self.scenarios:
            node = self.supervisor.getFromDef(scenario["def_name"])

            if node is None:
                print(f"Aviso: obstáculo não encontrado e será ignorado: {scenario['def_name']}")
                continue

            translation_field = node.getField("translation")
            current_position = np.array(
                translation_field.getSFVec3f(),
                dtype=np.float32
            )

            walk_direction = np.array(
                scenario["walk_direction"],
                dtype=np.float32
            )
            walk_direction = walk_direction / (np.linalg.norm(walk_direction) + 1e-8)

            obstacle = {
                "node": node,
                "translation_field": translation_field,
                "start": current_position.copy(),
                "end": current_position + walk_direction * scenario["walk_distance"],
                "trigger_position": current_position.copy(),
                "walk_direction": walk_direction,
                **scenario
            }

            self.obstacles.append(obstacle)

            print(f"Obstáculo carregado: {scenario['def_name']}")
            print(f"Posição inicial: {obstacle['start']}")
            print(f"Posição final: {obstacle['end']}")

    def reset(self):
        for obstacle in self.obstacles:
            obstacle["translation_field"].setSFVec3f(obstacle["start"].tolist())
            obstacle["node"].resetPhysics()
            obstacle["active"] = False
            obstacle["completed"] = False

    def step(self):
        vehicle_position = np.array(
            self.vehicle_translation_field.getSFVec3f(),
            dtype=np.float32
        )

        for obstacle in self.obstacles:
            if obstacle["completed"]:
                continue

            distance_to_pedestrian = np.linalg.norm(
                vehicle_position[[0, 1]] - obstacle["trigger_position"][[0, 1]]
            )

            if (
                not obstacle["active"]
                and distance_to_pedestrian <= obstacle["trigger_distance"]
            ):
                print("Pedestre ativado!")
                obstacle["active"] = True

            if obstacle["active"]:
                self._move_obstacle(obstacle)

    def _move_obstacle(self, obstacle):
        current_position = np.array(
            obstacle["translation_field"].getSFVec3f(),
            dtype=np.float32
        )

        end_position = obstacle["end"]

        direction = end_position - current_position
        distance = np.linalg.norm(direction)

        if distance < 0.05:
            obstacle["translation_field"].setSFVec3f(end_position.tolist())
            obstacle["node"].resetPhysics()
            obstacle["active"] = False
            obstacle["completed"] = True
            print("Pedestre terminou a travessia.")
            return

        direction = direction / (distance + 1e-8)
        new_position = current_position + direction * obstacle["speed"]

        obstacle["translation_field"].setSFVec3f(new_position.tolist())