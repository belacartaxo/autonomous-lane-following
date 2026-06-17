# Action limits
ACTION_LOW = [-0.5, -0.3]
ACTION_HIGH = [0.5, 0.6]

# Vehicle
MAX_VELOCITY = 30.0

# Episode limits
MAX_EPISODE_STEPS = 5000
MAX_LOST_LINE_STEPS = 300

# Stuck detection
STUCK_STEP_LIMIT = 500
STUCK_DISTANCE_THRESHOLD = 0.001

# Obstacle distances
OBSTACLE_NEAR_DISTANCE = 10.0
OBSTACLE_CLOSE_DISTANCE = 6.0
OBSTACLE_VERY_CLOSE_DISTANCE = 4.0
COLLISION_DISTANCE = 1.1

# Line detection
LINE_SIDE_THRESHOLD = 0.15
LINE_CENTER_THRESHOLD = 0.15
RECOVERY_CENTER_THRESHOLD = 0.25

# Normal lane following rewards
CENTER_REWARD_WEIGHT = 0.35
FORWARD_REWARD_WEIGHT = 0.12
LANE_ERROR_PENALTY = 0.30
STEERING_SMOOTHNESS_PENALTY = 0.04
CENTERED_LINE_BONUS = 0.20

# Obstacle avoidance rewards
AVOIDANCE_STEERING_REWARD = 0.35
AVOIDANCE_WRONG_STEERING_PENALTY = 0.35
AVOIDANCE_LINE_CENTER_WEIGHT = 0.05
AVOIDANCE_FORWARD_REWARD = 0.05

# Speed control near obstacle
OBSTACLE_SPEED_PENALTY = 0.30
OBSTACLE_SLOW_SPEED_TARGET = 0.35
OBSTACLE_SLOW_SPEED_REWARD = 0.20

# Very close obstacle
VERY_CLOSE_OBSTACLE_PENALTY = 0.70
VERY_CLOSE_SPEED_PENALTY = 0.30

# Recovery rewards
RECOVERY_SAME_SIDE_BONUS = 0.45
RECOVERY_CENTER_BONUS = 0.45
RECOVERY_PARTIAL_BONUS = 0.20
RECOVERY_LINE_LOST_PENALTY = 0.10
RECOVERY_STEERING_REWARD = 0.25
RECOVERY_WRONG_STEERING_PENALTY = 0.25
RECOVERY_SPEED_PENALTY = 0.10
RECOVERY_NO_SIDE_STEERING_REWARD = 0.10

# General penalties
LINE_LOST_PENALTY = 0.40
LINE_LOST_SPEED_PENALTY = 0.15
REVERSE_PENALTY = 0.10
STEP_PENALTY = 0.005
LOST_LINE_DONE_PENALTY = 1.0
STUCK_DONE_PENALTY = 1.0
COLLISION_PENALTY = 1.0

# Lane following steering
LANE_STEERING_REWARD = 0.20
LANE_WRONG_STEERING_PENALTY = 0.30
LANE_LOW_STEERING_PENALTY = 0.20
LANE_HIGH_ERROR_SPEED_PENALTY = 0.20

# Recovery scaling
RECOVERY_CENTER_WEIGHT = 0.40

# ─── Critical dynamic obstacle (fase ativa: peão/carro em movimento) ──────────

# Máximo de steps sem parar antes de terminar o episódio
CRITICAL_MAX_NOT_STOPPED_STEPS = 250

CRITICAL_FULL_STOP_SPEED_THRESHOLD = 0.005
CRITICAL_FORWARD_THROTTLE_THRESHOLD = 0.05

# Recompensa forte por parar — tem de superar claramente os incentivos de movimento
# O agente recebe isto por step enquanto estiver parado com obstáculo ativo
CRITICAL_STOP_REWARD = 0.60

# Penalização por não estar parado com obstáculo ativo
CRITICAL_NOT_STOPPED_PENALTY = 0.45

# Penalização extra se der throttle para a frente com obstáculo ativo
CRITICAL_FORWARD_PENALTY_WEIGHT = 0.50

# Penalização se estiver parado mas der throttle (pé no travão + acelerador)
CRITICAL_STOPPED_WITH_THROTTLE_PENALTY = 0.30

CRITICAL_RESUME_FORWARD_REWARD = 0.12
CRITICAL_UNNECESSARY_STOP_PENALTY = 0.20

# Penalização terminal por não ter parado a tempo
CRITICAL_FAILED_TO_STOP_PENALTY = 1.0
# Penalização terminal por colisão com obstáculo crítico
CRITICAL_COLLISION_PENALTY = 1.0

# ─── Fase inativa (sem obstáculo ativo: agente deve retomar a marcha) ─────────
# ATENÇÃO: estes valores têm de ser MENORES do que os da fase ativa.
# Se forem demasiado altos, o agente aprende que "andar é sempre melhor"
# e nunca para, mesmo com obstáculo ativo.

# Movimento mínimo esperado por step
CRITICAL_INACTIVE_MIN_MOVEMENT = 0.015

# Penalização moderada por não se mover — NÃO pode ser maior que CRITICAL_STOP_REWARD
CRITICAL_INACTIVE_LOW_MOVEMENT_PENALTY = 0.25

# Penalização moderada por não dar throttle
CRITICAL_INACTIVE_NO_THROTTLE_PENALTY = 0.20

# Bónus moderado por throttle positivo
CRITICAL_INACTIVE_FORWARD_BONUS_WEIGHT = 0.20

# Penalização por marcha atrás desnecessária
CRITICAL_INACTIVE_REVERSE_PENALTY_WEIGHT = 0.30

# Bónus único dado no momento exato em que o obstáculo desaparece
CRITICAL_RESUME_BONUS = 0.80

# ─── Timeout de inatividade pós-obstáculo ─────────────────────────────────────
# IMPORTANTE: este valor tem de ser suficientemente largo para não disparar
# durante a travessia legítima do obstáculo. O peão demora ~150 steps a
# atravessar. 300 steps dá margem suficiente.
CRITICAL_MAX_INACTIVE_STOPPED_STEPS = 300

# Penalização terminal quando o timeout de inatividade é atingido
CRITICAL_INACTIVE_TIMEOUT_PENALTY = 1.5

# Para normalizar a observação de idle time (deve ser igual ao timeout)
CRITICAL_INACTIVE_STOPPED_OBS_MAX = 300

CRITICAL_ACTIVE_FORWARD_PENALTY_BASE = 0.35
CRITICAL_ACTIVE_FORWARD_PENALTY_WEIGHT = 0.45
CRITICAL_ACTIVE_REVERSE_PENALTY_BASE = 0.15
CRITICAL_ACTIVE_REVERSE_PENALTY_WEIGHT = 0.20
CRITICAL_ACTIVE_STEERING_PENALTY_WEIGHT = 0.35
CRITICAL_ACTIVE_MOVEMENT_PENALTY_WEIGHT = 0.80
CRITICAL_ACTIVE_STOPPED_BONUS = 0.20
CRITICAL_ACTIVE_STEERING_THRESHOLD = 0.05
