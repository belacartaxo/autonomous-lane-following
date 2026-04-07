# Explicação do Código: main_controller.py

Este arquivo implementa um ambiente de simulação para um veículo autônomo usando o Gymnasium (uma extensão do OpenAI Gym) integrado com o Webots. O ambiente é projetado para treinamento de agentes de aprendizado por reforço (RL) para seguir pistas autônomas.

## Estrutura Geral

A classe principal é `WebotsVehicleEnv`, que herda de `gym.Env`. Ela representa um ambiente onde um agente pode controlar um veículo BMW X5 em um mundo simulado do Webots.

## Funções Principais

### `__init__(self)`
- Inicializa o supervisor do Webots.
- Configura dispositivos: LiDAR (Sick LMS 291) e câmera.
- Define espaços de ação e observação.
- Configura atuadores: direção e rodas.

**Espaço de Ação**: Uma caixa contínua com 2 valores:
- Direção: -0.5 a 0.5 (ângulo de esterçamento)
- Aceleração/Freio: -1.0 a 1.0 (velocidade alvo)

**Espaço de Observação**: Um dicionário com:
- LiDAR: 72 raios (distâncias de 0 a 10 metros)
- Câmera: Imagem RGB 64x64x3 (0-255)

### `reset(self, seed=None, options=None)`
- Reseta a posição e rotação do veículo para os valores iniciais.
- Reseta a física para parar qualquer movimento residual.
- Retorna as observações iniciais.

### `get_camera_image(self)`
- Processa os dados brutos da câmera do Webots.
- Converte de BGRA para RGB, removendo o canal alfa.
- Retorna um array numpy (altura, largura, 3).

### `step(self, action)`
- Aplica a ação ao veículo.
- Avança um passo na simulação.
- Coleta observações.
- Calcula recompensa e verifica término.
- Retorna: observações, recompensa, terminado, truncado, info.

### `_apply_action(self, action)`
- Define o ângulo de direção para ambos os lados (esquerdo e direito).
- Define a velocidade das rodas baseada na aceleração (escalada por 50.0).

### `_get_observations(self)`
- Coleta dados do LiDAR (substituindo NaN por 10.0).
- Obtém imagem da câmera.
- Retorna dicionário com LiDAR e câmera.

### `_compute_reward(self, obs, action)`
Calcula a recompensa baseada em:
- **Velocidade para frente**: Recompensa positiva por aceleração positiva (até 0.5).
- **Desvio lateral**: Penaliza diferença entre distâncias esquerda/direita do LiDAR (até 0.3).
- **Penalidade de tempo**: -0.01 por passo para desencorajar inatividade.
- **Colisão**: -20.0 se distância mínima do LiDAR < 0.45m, terminando o episódio.

## Bloco Principal
- Cria uma instância do ambiente.
- Reseta e imprime as formas das observações para verificação.

Este ambiente permite treinar agentes RL para dirigir veículos autônomos, focando em manter o centro da pista e evitar colisões.