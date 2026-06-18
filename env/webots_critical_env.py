import numpy as np
from gymnasium import spaces
from webots_setup import setup_webots_path 
setup_webots_path()

import configs.env_config as cfg

from env.webots_env import WebotsVehicleEnv
from env.critical_obstacles import CriticalObstacleManager


class WebotsCriticalVehicleEnv(WebotsVehicleEnv):
    """
    Webots environment with critical dynamic obstacles.

    The agent must fully stop when a critical obstacle is active,
    then resume normal lane following after the obstacle finishes moving.
    """

    def __init__(self):
        super().__init__()

        self.observation_space.spaces["critical_obstacle_active"] = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.critical_obstacles = CriticalObstacleManager(
            supervisor=self.robot,
            vehicle_translation_field=self.translation_field,
        )

        self.not_stopped_step_count = 0
        self.max_not_stopped_steps = cfg.CRITICAL_MAX_NOT_STOPPED_STEPS

        self.full_stop_speed_threshold = cfg.CRITICAL_FULL_STOP_SPEED_THRESHOLD
        self.forward_throttle_threshold = cfg.CRITICAL_FORWARD_THROTTLE_THRESHOLD

        self.stop_reward = cfg.CRITICAL_STOP_REWARD
        self.not_stopped_penalty = cfg.CRITICAL_NOT_STOPPED_PENALTY
        self.forward_penalty_weight = cfg.CRITICAL_FORWARD_PENALTY_WEIGHT

        self.stopped_with_throttle_penalty = (
            cfg.CRITICAL_STOPPED_WITH_THROTTLE_PENALTY
        )

        self.failed_to_stop_penalty = cfg.CRITICAL_FAILED_TO_STOP_PENALTY
        self.critical_collision_penalty = cfg.CRITICAL_COLLISION_PENALTY

        # Extra reward parameters for the non-critical phase.
        self.inactive_min_movement = getattr(
            cfg,
            "CRITICAL_INACTIVE_MIN_MOVEMENT",
            0.015,
        )
        self.inactive_low_movement_penalty = getattr(
            cfg,
            "CRITICAL_INACTIVE_LOW_MOVEMENT_PENALTY",
            0.60,
        )
        self.inactive_no_throttle_penalty = getattr(
            cfg,
            "CRITICAL_INACTIVE_NO_THROTTLE_PENALTY",
            0.40,
        )
        self.inactive_forward_bonus_weight = getattr(
            cfg,
            "CRITICAL_INACTIVE_FORWARD_BONUS_WEIGHT",
            0.35,
        )
        self.inactive_reverse_penalty_weight = getattr(
            cfg,
            "CRITICAL_INACTIVE_REVERSE_PENALTY_WEIGHT",
            0.45,
        )

        # [NOVO] Bónus único dado no momento em que o obstáculo desaparece
        self.resume_bonus = getattr(cfg, "CRITICAL_RESUME_BONUS", 0.80)

        self.previous_critical_position = np.array(
            self.translation_field.getSFVec3f(),
            dtype=np.float32,
        )

        # [NOVO] Flags de transição de estado
        self.was_obstacle_active = False
        self.resume_reward_given = False

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        self.critical_obstacles.reset()
        self.not_stopped_step_count = 0

        # [NOVO] Resetar flags de transição
        self.was_obstacle_active = False
        self.resume_reward_given = False

        self.previous_critical_position = np.array(
            self.translation_field.getSFVec3f(),
            dtype=np.float32,
        )

        obs["critical_obstacle_active"] = np.array(
            [0.0],
            dtype=np.float32,
        )

        return obs, info

    def _get_observations(self):
        obs = super()._get_observations()

        obs["critical_obstacle_active"] = np.array(
            [float(self.critical_obstacles.is_any_obstacle_active())],
            dtype=np.float32,
        )

        return obs

    def _get_vehicle_movement(self):
        current_position = np.array(
            self.translation_field.getSFVec3f(),
            dtype=np.float32,
        )

        movement = np.linalg.norm(
            current_position - self.previous_critical_position
        )

        self.previous_critical_position = current_position

        return float(movement)

    def _compute_reward(self, obs, action):
        throttle = float(action[1])
        steering = float(action[0])

        dynamic_obstacle_active = bool(obs["critical_obstacle_active"][0])

        vehicle_movement = self._get_vehicle_movement()

        if dynamic_obstacle_active:
            # [NOVO] Quando o obstáculo ativa de novo, reset do bónus de retoma
            self.resume_reward_given = False

            self.lost_line_steps = 0

            reward = -cfg.STEP_PENALTY
            done = False
            termination_reason = None

            vehicle_fully_stopped = (
                vehicle_movement <= self.full_stop_speed_threshold
            )

            _, collision_ray_count = self._get_collision_lidar_info(obs["lidar"])

            if collision_ray_count >= 3:
                self.collision_step_count += 1
            else:
                self.collision_step_count = 0

            if self.collision_step_count >= self.collision_step_limit:
                reward -= self.critical_collision_penalty
                done = True
                termination_reason = "collision_with_critical_obstacle"

                # [NOVO] Atualizar flag antes de sair
                self.was_obstacle_active = dynamic_obstacle_active
                return float(reward), done, termination_reason

            if vehicle_fully_stopped:
                reward += self.stop_reward
                self.not_stopped_step_count = 0

                if abs(throttle) > self.forward_throttle_threshold:
                    reward -= self.stopped_with_throttle_penalty

            else:
                reward -= self.not_stopped_penalty
                self.not_stopped_step_count += 1

                if abs(throttle) > self.forward_throttle_threshold:
                    reward -= abs(throttle) * self.forward_penalty_weight

            reward -= abs(steering) * cfg.STEERING_SMOOTHNESS_PENALTY

            if self.not_stopped_step_count >= self.max_not_stopped_steps:
                reward -= self.failed_to_stop_penalty
                done = True
                termination_reason = "failed_to_fully_stop_for_critical_obstacle"

            if self.stuck_step_count >= self.stuck_step_limit:
                reward -= cfg.STUCK_DONE_PENALTY
                done = True
                termination_reason = "stuck"

            self.current_step += 1

            if self.current_step >= self.max_episode_steps:
                done = True
                termination_reason = "max_episode_steps"

            # [NOVO] Atualizar flag de transição
            self.was_obstacle_active = dynamic_obstacle_active

            return float(reward), done, termination_reason

        # --- Fase sem obstáculo crítico ---

        self.not_stopped_step_count = 0

        # [NOVO] Detetar transição: obstáculo acabou de desaparecer
        just_cleared = self.was_obstacle_active and not dynamic_obstacle_active
        if just_cleared and not self.resume_reward_given:
            print("Obstáculo passou — bónus de retoma dado ao agente.")
            # Este bónus é dado uma única vez no momento da transição
            # para quebrar a inércia de paragem aprendida
            resume_bonus = self.resume_bonus
            self.resume_reward_given = True
        else:
            resume_bonus = 0.0

        # [NOVO] Atualizar flag de transição
        self.was_obstacle_active = dynamic_obstacle_active

        reward, done, termination_reason = super()._compute_reward(
            obs,
            action,
        )

        # Aplicar bónus de retoma (único, no primeiro step após o obstáculo)
        reward += resume_bonus

        # Quando não há obstáculo crítico, o agente deve mover-se para a frente.
        # Penalizações fortes para quebrar o hábito de ficar parado.
        if vehicle_movement < self.inactive_min_movement:
            reward -= self.inactive_low_movement_penalty

        # Penalizar throttle fraco quando não há razão para parar.
        if throttle <= self.forward_throttle_threshold:
            reward -= self.inactive_no_throttle_penalty

        # Recompensar throttle positivo para incentivar retoma da faixa.
        if throttle > self.forward_throttle_threshold:
            reward += throttle * self.inactive_forward_bonus_weight

        # Penalizar marcha atrás fora de situações de obstáculo crítico.
        if throttle < -self.forward_throttle_threshold:
            reward -= abs(throttle) * self.inactive_reverse_penalty_weight

        return float(reward), done, termination_reason

    def step(self, action):
        real_action = self._denormalize_action(action)

        self._apply_action(real_action)

        self.critical_obstacles.step()

        if self.robot.step(self.timestep) == -1:
            return {}, 0.0, True, False, {}

        obs = self._get_observations()

        self._check_if_is_stuck(real_action)

        reward, done, termination_reason = self._compute_reward(
            obs,
            real_action,
        )

        min_front_lidar, close_front_ray_count = self._get_front_lidar_info(
            obs["lidar"]
        )

        min_lidar_distance, collision_ray_count = self._get_collision_lidar_info(
            obs["lidar"]
        )

        critical_obstacle_active = bool(obs["critical_obstacle_active"][0])

        collision = bool(
            termination_reason in [
                "collision",
                "collision_with_critical_obstacle",
            ]
        )

        return obs, float(reward), done, False, {
            "collision": collision,
            "termination_reason": termination_reason,
            "critical_obstacle_active": critical_obstacle_active,
            "not_stopped_step_count": self.not_stopped_step_count,
            "max_not_stopped_steps": self.max_not_stopped_steps,
            "min_front_lidar": min_front_lidar,
            "close_front_rays": close_front_ray_count,
            "min_lidar_distance": min_lidar_distance,
            "collision_rays": collision_ray_count,
            "lost_line_steps": self.lost_line_steps,
            "stuck_step_count": self.stuck_step_count,
            "collision_step_count": self.collision_step_count,
            # [NOVO] Info útil para debug no tensorboard
            "resume_reward_given": self.resume_reward_given,
            "was_obstacle_active": self.was_obstacle_active,
        }
