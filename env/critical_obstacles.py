import numpy as np


class CriticalObstacleManager:
    """
    Manager for critical dynamic obstacle scenarios.

    Supports pedestrians and vehicles that start from their current position
    in the Webots world and move when the ego vehicle enters a 2D trigger radius.
    """

    def __init__(self, supervisor, vehicle_translation_field):
        self.supervisor = supervisor
        self.vehicle_translation_field = vehicle_translation_field

        self.scenarios = [
            {
                "def_name": "PEDESTRIAN_1",
                "label": "Pedestre",
                "trigger_distance": 25.0,
                "move_direction": [0.0, 1.0, 0.0],
                "move_distance": 20.0,
                "speed": 0.08,
                "active": False,
                "completed": False,
            },
            {
                "def_name": "VEHICLE_1",
                "label": "Automóvel",
                "trigger_distance": 30.0,
                "move_direction": [1.0, 0.0, 0.0],
                "move_distance": 30.0,
                "speed": 0.15,
                "active": False,
                "completed": False,
            },
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

            move_direction = np.array(
                scenario["move_direction"],
                dtype=np.float32
            )
            move_direction = move_direction / (np.linalg.norm(move_direction) + 1e-8)

            obstacle = {
                "node": node,
                "translation_field": translation_field,
                "start": current_position.copy(),
                "end": current_position + move_direction * scenario["move_distance"],
                "trigger_position": current_position.copy(),
                "move_direction": move_direction,
                **scenario
            }

            self.obstacles.append(obstacle)

            print(f"Obstáculo carregado: {scenario['label']} ({scenario['def_name']})")
            print(f"Posição inicial: {obstacle['start']}")
            print(f"Posição final: {obstacle['end']}")

    def reset(self):
        for obstacle in self.obstacles:
            obstacle["translation_field"].setSFVec3f(obstacle["start"].tolist())
            obstacle["node"].resetPhysics()
            obstacle["active"] = False
            obstacle["completed"] = False

    def step(self):
        ego_position = np.array(
            self.vehicle_translation_field.getSFVec3f(),
            dtype=np.float32
        )

        for obstacle in self.obstacles:
            if obstacle["completed"]:
                continue

            distance_to_obstacle = np.linalg.norm(
                ego_position[[0, 1]] - obstacle["trigger_position"][[0, 1]]
            )

            if (
                not obstacle["active"]
                and distance_to_obstacle <= obstacle["trigger_distance"]
            ):
                print(f"{obstacle['label']} ativado!")
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
            print(f"{obstacle['label']} terminou o movimento.")
            return

        direction = direction / (distance + 1e-8)
        new_position = current_position + direction * obstacle["speed"]

        obstacle["translation_field"].setSFVec3f(new_position.tolist())

    def is_any_obstacle_active(self):
        """Return True if any critical obstacle is currently moving."""
        return any(obstacle["active"] for obstacle in self.obstacles)