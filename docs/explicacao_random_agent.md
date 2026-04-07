# Explicação do Código: random_agent.py

Este arquivo é um script simples para testar o ambiente `WebotsVehicleEnv` usando um agente aleatório. Ele executa episódios de simulação onde ações são escolhidas aleatoriamente, permitindo avaliar o comportamento da recompensa e do ambiente.

## Estrutura Geral

O script importa o ambiente personalizado e executa um loop de testes com 3 episódios.

## Código Principal

### Importações
- `from main_controller import WebotsVehicleEnv`: Importa a classe do ambiente.
- `import numpy as np`: Embora não usado diretamente aqui, pode ser para extensões futuras.

### Inicialização
- `env = WebotsVehicleEnv()`: Cria uma instância do ambiente.

### Loop de Episódios
- Executa 3 episódios (`for episode in range(3)`).
- Para cada episódio:
  - Reseta o ambiente: `obs, _ = env.reset()`.
  - Inicializa variáveis: `done = False`, `truncated = False`, `total_reward = 0`, `step_count = 0`.
  - Loop interno enquanto não terminado:
    - Amostra uma ação aleatória: `action = env.action_space.sample()`.
    - Executa o passo: `obs, reward, done, truncated, info = env.step(action)`.
    - Acumula recompensa e conta passos.
    - A cada 10 passos, imprime status: episódio, passo, recompensa instantânea e total.
  - Ao fim do episódio, imprime resumo: passos totais e recompensa final.

## Propósito
Este script serve para testar o ambiente antes de implementar agentes de RL mais sofisticados. Ele verifica se o ambiente funciona corretamente, se as recompensas são calculadas adequadamente e se não há erros de simulação.