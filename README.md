# Autonomous Lane Following

This project implements a reinforcement learning environment for autonomous lane following using Webots simulation and Gymnasium. The goal is to train an agent to control a vehicle (BMW X5) to stay centered on the lane while avoiding collisions.

## Prerequisites

- [Webots](https://cyberbotics.com/doc/guide/installation-procedure) (download and install)
- Python 3.8+
- Virtual environment support

## Installation

1. Clone or download this repository.

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Add Webots Python path to your system PATH (refer to Webots documentation for your OS).

## Usage

1. Open Webots and load the world file: `worlds/city_default.wbt` (File > Open World).

2. In your IDE or terminal, run the random agent test:
   ```bash
   python controllers/main_controller/random_agent.py
   ```

This will run 3 episodes with a random agent to test the environment and reward system.

## Project Structure

- `controllers/main_controller/`: Main environment and agent code
  - `main_controller.py`: Gymnasium environment wrapper for Webots vehicle simulation
  - `random_agent.py`: Test script with random actions
- `worlds/`: Webots world files
  - `city_default.wbt`: Default city simulation world
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Environment Details

The `WebotsVehicleEnv` class provides:
- **Action Space**: Continuous 2D vector [steering (-0.5 to 0.5), throttle/brake (-1.0 to 1.0)]
- **Observation Space**: Dictionary with LiDAR (72 rays) and camera (64x64 RGB)
- **Reward Function**: Encourages forward speed, penalizes lateral deviation and collisions

## License

This project is for educational purposes. 