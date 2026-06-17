import numpy as np


class CriticalObstacleManager:
    """
    Manager for critical dynamic obstacle scenarios.

    Supports pedestrians and vehicles that start from their current position
    in the Webots world and move when the ego vehicle enters a 2D trigger radius.

    ALTERAÇÕES vs original:
    - Peão: speed 0.01 → 0.05 (5x mais rápido).
      Com speed=0.01 o peão demora 2000 steps a atravessar (20m / 0.01).
      Com speed=0.05 demora 400 steps — mais realista e o agente tem sinal
      claro de "peão a atravessar" sem precisar de estar parado 2000 steps.
    - Carro: speed 0.15 → 0.20 (ligeiramente mais rápido, mais visível).
    - trigger_distance do peão: 25.0 → 20.0 para o agente ver o peão
      no LiDAR antes do trigger (OBSTACLE_NEAR_DISTANCE=10.0 não chegava).
    """

    def __init__(self, supervisor, vehicle_translation_field):
        self.supervisor = supervisor
        self.vehicle_translation_field = vehicle_translation_field

        self.scenarios = [
            {
                "def_name": "PEDESTRIAN_1",
                "label": "Pedestre",
                "trigger_distance": 20.0,   # reduzido de 25→20: trigger mais perto
                "move_direction": [0.0, 1.0, 0.0],
                "move_distance": 20.0,
                "speed": 0.05,              # aumentado de 0.01→0.05: 400 steps em vez de 2000
                "active": False,
                "completed": False,
            },
            {
                "def_name": "VEHICLE_1",
                "label": "Automóvel",
                "trigger_distance": 30.0,
                "move_direction": [1.0, 0.0, 0.0],
                "move_distance": 30.0,
                "speed": 0.20,              # ligeiramente aumentado de 0.15→0.20
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
            print(f"  Posição inicial: {obstacle['start']}")
            print(f"  Posição final:   {obstacle['end']}")
            print(f"  Speed:           {scenario['speed']} u/step")
            print(f"  Steps estimados: {int(scenario['move_distance'] / scenario['speed'])}")

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
                print(f"{obstacle['label']} ativado! Distância ao ego: {distance_to_obstacle:.1f}m")
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
