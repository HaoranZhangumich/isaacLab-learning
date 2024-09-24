# isaacLab-learning
Documentation for self-learning  isaacLab

## offical web 
[IsaacLab](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html)


### Manager Based environment
In Isaac Lab, manager-based environments are implemented as **envs.ManagerBasedEnv** and **envs.ManagerBasedRLEnv** classes. The two classes are very similar, but envs.ManagerBasedRLEnv is useful for **reinforcement learning** tasks and contains rewards, terminations, curriculum and command generation. The **envs.ManagerBasedEnv** class is useful for traditional robot control and doesn’t contain rewards and terminations.

#### import env
```python

from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg

```

#### envs.ManagerBasedEnv
The base class **envs.ManagerBasedEnv** wraps around many intricacies of the simulation interaction and provides a simple interface for the user to run the simulation and interact with it.
1. **scene.InteractiveScene** - The scene that is used for the simulation.

2. **managers.ActionManager** - The manager that handles actions.

3. **managers.ObservationManager** - The manager that handles observations.

4. **managers.EventManager** - The manager that schedules operations (such as domain randomization) at specified simulation events. For instance, at startup, on resets, or periodic intervals.

#### Scene
refer to Interactive Scene

#### Action
1. Used **managers.ActionManager** instead of assets.Articulation.set_joint_effort_target()
2. Each action term is responsible for applying control over a specific aspect of the environment. eg:  for robotic arm, we can have two action terms – one for controlling the joints of the arm, and the other for controlling the gripper. This composition allows the user to define different control schemes for different aspects of the environment.

```python
@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)
```

#### Observation
Use **manager.ObservationManager** and similar to action manager, observation manager also comprise of multiple observation terms (observation groups used to define different observation spaces)
eg.  for hierarchical control, we may want to define two observation groups – one for the low level controller and the other for the high level controller. It is assumed that all the observation terms in a group have the same dimensions.

```python
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
```
1. **managers.ObservationGroupCfg** collects different observation terms and help define common properties for the group, such as enabling noise corruption or concatenating the observations into a single tensor.

2. **managers.ObservationTermCfg** takes in the **managers.ObservationTermCfg.func** that specifies the function or callable class that computes the observation for that term. It includes other parameters for defining the noise model, clipping, scaling and etc.
```python
@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
```

#### Events
1. **managers.EventManager** class is responsible for events corresponding to changes in the simulation state.This includes resetting (or randomizing) the scene, randomizing physical properties (such as mass, friction, etc.), and varying visual properties (such as colors, textures, etc.)

```python
from omni.isaac.lab.managers import EventTermCfg as EventTerm
```

2. Each of these are specified through the **managers.EventTermCfg** class, which takes in the **managers.EventTermCfg.func** that specifies the function or callable class that performs the event.

3. it expects **mode** of event. mode can be own defined and **ManagerBasedEnv** provides three commonly used modes:
- "startup" - Event that takes place only once at environment startup.

- "reset" - Event that occurs on environment termination and reset.

- "interval" - Event that are executed at a given interval, i.e., periodically after a certain number of steps.


```python
@configclass
class EventCfg:
    """Configuration for events."""

    # on startup
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add",
        },
    )

    # on reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )
```

#### All together (CartpoleEnvCfg)
```python 
@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = CartpoleSceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz
```

#### In simulation
1. **envs.ManagerBasedEnv.reset()** reset the environment
2. **envs.ManagerBasedEnv.step()** step the environment
3. **envs.ManagerBasedEnv** did not have notions of terminations since that concept is specific for episodic tasks and user have to define it
4. An important thing to note above is that the entire simulation loop is wrapped inside the **torch.inference_mode()** context manager. This is because the environment uses PyTorch operations under-the-hood and we want to ensure that the simulation is not slowed down by the overhead of PyTorch’s autograd engine and gradients are not computed for the simulation operations.

```python
def main():
    """Main function."""
    # parse the arguments
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, _ = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()
```