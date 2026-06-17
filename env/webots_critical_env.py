import numpy as np
from gymnasium import spaces

import configs.env_config as cfg

from env.webots_env import WebotsVehicleEnv
from env.critical_obstacles import CriticalObstacleManager


class WebotsCriticalVehicleEnv(WebotsVehicleEnv):
    """
    Webots environment with critical dynamic obstacles.

    Comportamento esperado:
      - Obstáculo ativo  → agente PARA completamente (throttle ≈ 0, velocidade ≈ 0)
      - Obstáculo inativo → agente retoma lane following normalmente

    Princípio de design dos rewards:
      CRITICAL_STOP_REWARD (0.60/step parado com obstáculo) deve ser MAIOR
      do que qualquer incentivo de movimento na fase inativa (0.20-0.25/step).
      Assim o agente nunca prefere andar a parar quando há obstáculo.

      O timeout de inatividade (300 steps) é largo o suficiente para não
      interferir com paragens legítimas para o peão.
    """

    def __init__(self):
        super().__init__()

        # Observação: obstáculo crítico ativo?
        self.observation_space.spaces["critical_obstacle_active"] = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        # Observação: há quanto tempo está parado sem obstáculo (0→1)
        # Dá ao agente contexto temporal para distinguir
        # "acabei de parar para o peão" de "estou parado sem razão"
        self.observation_space.spaces["inactive_stopped_steps_norm"] = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        self.critical_obstacles = CriticalObstacleManager(
            supervisor=self.robot,
            vehicle_translation_field=self.translation_field,
        )

        # ── Parâmetros fase crítica ────────────────────────────────────────────
        self.not_stopped_step_count = 0
        self.max_not_stopped_steps = cfg.CRITICAL_MAX_NOT_STOPPED_STEPS
        self.full_stop_speed_threshold = cfg.CRITICAL_FULL_STOP_SPEED_THRESHOLD
        self.forward_throttle_threshold = cfg.CRITICAL_FORWARD_THROTTLE_THRESHOLD
        self.stop_reward = cfg.CRITICAL_STOP_REWARD
        self.not_stopped_penalty = cfg.CRITICAL_NOT_STOPPED_PENALTY
        self.forward_penalty_weight = cfg.CRITICAL_FORWARD_PENALTY_WEIGHT
        self.stopped_with_throttle_penalty = cfg.CRITICAL_STOPPED_WITH_THROTTLE_PENALTY
        self.failed_to_stop_penalty = cfg.CRITICAL_FAILED_TO_STOP_PENALTY
        self.critical_collision_penalty = cfg.CRITICAL_COLLISION_PENALTY

        # ── Parâmetros fase inativa (sem obstáculo) ────────────────────────────
        self.inactive_min_movement = getattr(cfg, "CRITICAL_INACTIVE_MIN_MOVEMENT", 0.015)
        self.inactive_low_movement_penalty = getattr(cfg, "CRITICAL_INACTIVE_LOW_MOVEMENT_PENALTY", 0.25)
        self.inactive_no_throttle_penalty = getattr(cfg, "CRITICAL_INACTIVE_NO_THROTTLE_PENALTY", 0.20)
        self.inactive_forward_bonus_weight = getattr(cfg, "CRITICAL_INACTIVE_FORWARD_BONUS_WEIGHT", 0.20)
        self.inactive_reverse_penalty_weight = getattr(cfg, "CRITICAL_INACTIVE_REVERSE_PENALTY_WEIGHT", 0.30)
        self.resume_bonus = getattr(cfg, "CRITICAL_RESUME_BONUS", 0.80)

        # ── Timeout de inatividade ─────────────────────────────────────────────
        self.max_inactive_stopped_steps = getattr(cfg, "CRITICAL_MAX_INACTIVE_STOPPED_STEPS", 300)
        self.inactive_timeout_penalty = getattr(cfg, "CRITICAL_INACTIVE_TIMEOUT_PENALTY", 1.5)
        self.inactive_stopped_obs_max = getattr(cfg, "CRITICAL_INACTIVE_STOPPED_OBS_MAX", 300)

        # ── Estado interno ─────────────────────────────────────────────────────
        self.previous_critical_position = np.array(
            self.translation_field.getSFVec3f(), dtype=np.float32)
        self.was_obstacle_active = False
        self.resume_reward_given = False
        self.inactive_stopped_step_count = 0

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        self.critical_obstacles.reset()
        self.not_stopped_step_count = 0
        self.inactive_stopped_step_count = 0
        self.was_obstacle_active = False
        self.resume_reward_given = False

        self.previous_critical_position = np.array(
            self.translation_field.getSFVec3f(), dtype=np.float32)

        obs["critical_obstacle_active"] = np.array([0.0], dtype=np.float32)
        obs["inactive_stopped_steps_norm"] = np.array([0.0], dtype=np.float32)
        return obs, info

    # ──────────────────────────────────────────────────────────────────────────
    def _get_observations(self):
        obs = super()._get_observations()

        obstacle_active = float(self.critical_obstacles.is_any_obstacle_active())
        obs["critical_obstacle_active"] = np.array([obstacle_active], dtype=np.float32)

        norm_idle = min(
            self.inactive_stopped_step_count / max(self.inactive_stopped_obs_max, 1),
            1.0,
        )
        obs["inactive_stopped_steps_norm"] = np.array([norm_idle], dtype=np.float32)
        return obs

    # ──────────────────────────────────────────────────────────────────────────
    def _get_vehicle_movement(self):
        current_pos = np.array(self.translation_field.getSFVec3f(), dtype=np.float32)
        movement = float(np.linalg.norm(current_pos - self.previous_critical_position))
        self.previous_critical_position = current_pos
        return movement

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_reward(self, obs, action):
        throttle = float(action[1])
        steering = float(action[0])
        dynamic_obstacle_active = bool(obs["critical_obstacle_active"][0])
        vehicle_movement = self._get_vehicle_movement()

        # ══════════════════════════════════════════════════════════════════════
        # FASE CRÍTICA — obstáculo em movimento → PARAR
        # ══════════════════════════════════════════════════════════════════════
        if dynamic_obstacle_active:
            # Reset de estado de transição
            self.resume_reward_given = False
            self.inactive_stopped_step_count = 0
            self.lost_line_steps = 0

            reward = -cfg.STEP_PENALTY
            done = False
            termination_reason = None

            vehicle_fully_stopped = vehicle_movement <= self.full_stop_speed_threshold

            _, collision_ray_count = self._get_collision_lidar_info(obs["lidar"])
            if collision_ray_count >= 3:
                self.collision_step_count += 1
            else:
                self.collision_step_count = 0

            if self.collision_step_count >= self.collision_step_limit:
                reward -= self.critical_collision_penalty
                done = True
                termination_reason = "collision_with_critical_obstacle"
                self.was_obstacle_active = dynamic_obstacle_active
                return float(reward), done, termination_reason

            if vehicle_fully_stopped:
                # PARADO com obstáculo: recompensa forte (0.60/step)
                reward += self.stop_reward
                self.not_stopped_step_count = 0
                # Penalizar se tentar acelerador mesmo parado
                if abs(throttle) > self.forward_throttle_threshold:
                    reward -= self.stopped_with_throttle_penalty
            else:
                # EM MOVIMENTO com obstáculo: penalização forte
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

            self.was_obstacle_active = dynamic_obstacle_active
            return float(reward), done, termination_reason

        # ══════════════════════════════════════════════════════════════════════
        # FASE INATIVA — sem obstáculo → RETOMAR lane following
        # ══════════════════════════════════════════════════════════════════════
        self.not_stopped_step_count = 0

        # Bónus único de transição: dado apenas no primeiro step após o obstáculo
        just_cleared = self.was_obstacle_active and not dynamic_obstacle_active
        if just_cleared and not self.resume_reward_given:
            print("Obstáculo passou — bónus de retoma aplicado.")
            resume_bonus = self.resume_bonus
            self.resume_reward_given = True
            self.inactive_stopped_step_count = 0
        else:
            resume_bonus = 0.0

        self.was_obstacle_active = dynamic_obstacle_active

        # Reward base do lane following (do env pai)
        reward, done, termination_reason = super()._compute_reward(obs, action)
        reward += resume_bonus

        # ── Incentivos/penalizações de movimento na fase inativa ───────────────
        # NOTA: estes valores são MODERADOS de propósito.
        # Se forem demasiado altos, competem com o CRITICAL_STOP_REWARD
        # e o agente nunca aprende a parar.
        vehicle_is_stopped = vehicle_movement < self.inactive_min_movement

        if vehicle_is_stopped:
            self.inactive_stopped_step_count += 1
            reward -= self.inactive_low_movement_penalty      # -0.25/step parado
        else:
            self.inactive_stopped_step_count = 0              # reset ao mover

        if throttle <= self.forward_throttle_threshold:
            reward -= self.inactive_no_throttle_penalty       # -0.20/step sem throttle

        if throttle > self.forward_throttle_threshold:
            reward += throttle * self.inactive_forward_bonus_weight  # bónus por andar

        if throttle < -self.forward_throttle_threshold:
            reward -= abs(throttle) * self.inactive_reverse_penalty_weight

        # ── Timeout de inatividade ─────────────────────────────────────────────
        # Só termina o episódio se o agente recusar MESMO mover-se durante
        # 300 steps seguidos sem qualquer obstáculo ativo.
        if self.inactive_stopped_step_count >= self.max_inactive_stopped_steps:
            reward -= self.inactive_timeout_penalty
            done = True
            termination_reason = "inactive_timeout_refused_to_resume"
            print(
                f"TIMEOUT DE INATIVIDADE: agente parado há "
                f"{self.inactive_stopped_step_count} steps sem obstáculo. Episódio terminado."
            )

        return float(reward), done, termination_reason

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action):
        real_action = self._denormalize_action(action)
        self._apply_action(real_action)
        self.critical_obstacles.step()

        if self.robot.step(self.timestep) == -1:
            return {}, 0.0, True, False, {}

        obs = self._get_observations()
        self._check_if_is_stuck(real_action)

        reward, done, termination_reason = self._compute_reward(obs, real_action)

        min_front_lidar, close_front_ray_count = self._get_front_lidar_info(obs["lidar"])
        min_lidar_distance, collision_ray_count = self._get_collision_lidar_info(obs["lidar"])
        critical_obstacle_active = bool(obs["critical_obstacle_active"][0])

        collision = bool(termination_reason in [
            "collision", "collision_with_critical_obstacle",
        ])

        return obs, float(reward), done, False, {
            "collision": collision,
            "termination_reason": termination_reason,
            "critical_obstacle_active": critical_obstacle_active,
            "not_stopped_step_count": self.not_stopped_step_count,
            "max_not_stopped_steps": self.max_not_stopped_steps,
            "inactive_stopped_step_count": self.inactive_stopped_step_count,
            "min_front_lidar": min_front_lidar,
            "close_front_rays": close_front_ray_count,
            "min_lidar_distance": min_lidar_distance,
            "collision_rays": collision_ray_count,
            "lost_line_steps": self.lost_line_steps,
            "stuck_step_count": self.stuck_step_count,
            "collision_step_count": self.collision_step_count,
            "resume_reward_given": self.resume_reward_given,
            "was_obstacle_active": self.was_obstacle_active,
        }
