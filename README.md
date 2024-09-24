# isaacLab-learning
Documentation for self-learning  isaacLab

## offical web 
https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html


### Manager Based environment
In Isaac Lab, manager-based environments are implemented as envs.ManagerBasedEnv and envs.ManagerBasedRLEnv classes. The two classes are very similar, but envs.ManagerBasedRLEnv is useful for reinforcement learning tasks and contains rewards, terminations, curriculum and command generation. The envs.ManagerBasedEnv class is useful for traditional robot control and doesn’t contain rewards and terminations.

#### import env
'''
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
'''

#### envs.ManagerBasedEnv
The base class envs.ManagerBasedEnv wraps around many intricacies of the simulation interaction and provides a simple interface for the user to run the simulation and interact with it.
1. scene.InteractiveScene - The scene that is used for the simulation.

2. managers.ActionManager - The manager that handles actions.

3. managers.ObservationManager - The manager that handles observations.

4. managers.EventManager - The manager that schedules operations (such as domain randomization) at specified simulation events. For instance, at startup, on resets, or periodic intervals.

#### Action
1. Used managers.ActionManager instead of assets.Articulation.set_joint_effort_target()
2. Each action term is responsible for applying control over a specific aspect of the environment. eg:  for robotic arm, we can have two action terms – one for controlling the joints of the arm, and the other for controlling the gripper. This composition allows the user to define different control schemes for different aspects of the environment.

'''
@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)
'''

#### Observation