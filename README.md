# 3D Robotics Simulator in OpenAI Gym Environment

### Authors: Tianyu Li (Anthony), Weizhuo Wang (Ken)

This is an OpenAI gym simulation environment designed for Reinforcement Learning(RL) agent training. The repo itself contains the common framework structure for RL training and the simulator. To use this repo, users should:
1. Enter the learning algorithm in Agent.py
2. Provide simulation rendering model in vpython. (An example of the gym_quadrotor enviroment is given, follow the format to create a new folder if you need to train other kinds of agent such as a mobile robot)

![Screenshot](sss.png)

###### Requirements:
- gym
- vpython
- matplotlib
- numpy

###### Running Instructions:

1. In root directory, activate python3 virtual environment
> source venv/bin/activate

2. Install Requirements Package (only for the first time)
> pip3 install -r requirements.txt

3. Run the gym environment
> python3 main.py



###### Important files:

|#| File Name          | Description     |
|-| ------------- |-------------|
|1| main.py    | Main running file |
|2| Agent.py    | RL network file |
|3| gym_quadrotor/envs/quadrotor_env.py | Simulator environment |
