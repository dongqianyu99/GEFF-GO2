# Galaxea A1 Simulation Isaac Sim Tutorial  

[official tutorial](http://galaxea.tech/Product_User_Guide/Guide/R1/Simulation_Isaac_Lab_Tutorial/#define-observation-action-reward-etc)  

## Installation

As you wish. First, install **Isaac Sim**.  

`git clone https://github.com/userguide-galaxea/Galaxea_Lab.git`  

**From now on, you can refer to the tutorial of** [Isaac Lab binary installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html#installing-isaac-lab).  

After installation, the robot works well (~stupidly~)   

![alt text](image.png)

## Code Reading  
### Minimal Excutable Code Unit  
Path: /source/standalone/galaxea/basic/simple_env.py
~~~py
"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # use openAI Gym as basic environment
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

def main():
    """Zero actions agent with Isaac Lab environment."""
    env_cfg = parse_env_cfg(
        "Isaac-R1-Lift-Bin-IK-Rel-Direct-v0",  # task_name
        use_gpu= True,
        num_envs= 1,
        use_fabric= True,  # Defines whether to use the current USD for I/O. The default value is True
    )
    # create environment
    env = gym.make("Isaac-R1-Lift-Bin-IK-Rel-Direct-v0", cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            if True:
                # sample actions from -1 to 1
                actions = (
                    0.05 * torch.rand(env.action_space.shape, device=env.unwrapped.device)
                )
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
~~~

### __init__.py
PATH: /source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/galaxea/direct/lift/__init__.py  
~~~py
##
# Register Gym environments.
##

gym.register(
    id="Isaac-R1-Multi-Fruit-IK-Abs-Direct-v0",  # unique identifier for the environment
    entry_point="omni.isaac.lab_tasks.galaxea.direct.lift:R1MultiFruitEnv",  # spacify the path to the implementation class of the environment
    disable_env_checker=True,  
    kwargs={
        "env_cfg_entry_point": R1MultiFruitAbsEnvCfg,  # spacific configuration
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FrankaCabinetPPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
~~~

### pick_fruit_env.py  
PATH: /source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/galaxea/direct/lift/pick_fruit_env.py  

~~~py
class R1MultiFruitEnv(DirectRLEnv):  # "DirectRLEnv" is a custom reinforcement learning environment
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_observations()
    #   |-- _get_rewards()
    #   |-- _get_dones()
    #   |-- _reset_idx(env_ids)

    cfg: (
        R1LiftCubeAbsEnvCfg
        | R1LiftCubeRelEnvCfg
        | R1LiftBinAbsEnvCfg
        | R1LiftBinRelEnvCfg
        | R1MultiFruitAbsEnvCfg
    )
~~~

#### __init__
~~~py
def __init__(
        self,
        cfg: (
            R1LiftCubeAbsEnvCfg
            | R1LiftCubeRelEnvCfg
            | R1LiftBinAbsEnvCfg
            | R1LiftBinRelEnvCfg
            | R1MultiFruitAbsEnvCfg
        ),
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)  # Calls the superclass's initializer (DirectRLEnv)

        # action type
        self.action_type = self.cfg.action_type

        # joint limits
        ...

        # track goal reset state
        ...

        # default goal pose, i.e. the init target pose in the robot base_link frame
        self.goal_rot = torch.zeros(
            (self.num_envs, 4), dtype=torch.float, device=self.device  # The rotation quaternion
        )
        self.goal_rot[:, 0] = 1.0  # default to be [1.0, 0.0, 0.0, 0.0] (without rotation)
        self.goal_pos = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device  # The position
        )
        self.goal_pos[:, :] = torch.tensor([0.4, 0.0, 1.2], device=self.device)

        # vis markers
        ...

        # end-effector offset w.r.t the *_arm_link6 frame
        ...

        # left/right arm/gripper joint ids
        self._setup_robot()
        # ik controller
        self._setup_ik_controller()

        self.succ = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # whether the job is successful
        self.dt = self.cfg.sim.dt * self.cfg.decimation  # simulation step size

        self.object_id = random.randint(0, 2)
        self.init_pos = torch.zeros(size=(self.num_envs, 3), device=self.device)

        print("R1LiftEnv is initialized. ActionType: ", self.action_type)
~~~

#### _setup_scene
~~~py
def _setup_scene(self):
        self._object = [0]*4 
        self._robot = Articulation(self.cfg.robot_cfg)
        self._drop_height = 0.96
        # add robot, object

        # add a carrot
        object1_cfg = copy.deepcopy(self.cfg.carrot_cfg)
        object1_pos = (0.35, -0.35, self._drop_height)  # initial pos
        object1_cfg.init_state.pos = object1_pos
        object1_cfg.spawn.scale = (0.3, 0.3, 0.3)
        self._object[0] = RigidObject(object1_cfg)
        self._object.append(RigidObject(object1_cfg))

        # add two bananas
        ...

        # add a basket
        ...

        # add table which is a static object
        if self.cfg.table_cfg.spawn is not None:
            self.cfg.table_cfg.spawn.scale = (0.09, 0.09, 0.09)
            self.cfg.table_cfg.spawn.func(
                self.cfg.table_cfg.prim_path,
                self.cfg.table_cfg.spawn,
                translation=self.cfg.table_cfg.init_state.pos,
                orientation=self.cfg.table_cfg.init_state.rot,
            )

        # add camera
        if self.cfg.enable_camera:
            self._front_camera = Camera(self.cfg.front_camera_cfg)
            self._left_wrist_camera = Camera(self.cfg.left_wrist_camera_cfg)
            self._right_wrist_camera = Camera(self.cfg.right_wrist_camera_cfg)
            self.scene.sensors["front_camera"] = self._front_camera
            self.scene.sensors["left_wrist_camera"] = self._left_wrist_camera
            self.scene.sensors["right_wrist_camera"] = self._right_wrist_camera

        # frame transformer
        self._left_ee_frame = FrameTransformer(self.cfg.left_ee_frame_cfg)
        self._right_ee_frame = FrameTransformer(self.cfg.right_ee_frame_cfg)

        # add to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object0"] = self._object[0]
        self.scene.rigid_objects["object1"] = self._object[1]
        self.scene.rigid_objects["object2"] = self._object[2]
        self.scene.rigid_objects["object3"] = self._object[3]
        self.scene.sensors["left_ee_frame"] = self._left_ee_frame
        self.scene.sensors["right_ee_frame"] = self._right_ee_frame
        self.scene.extras["table"] = XFormPrimView(
            self.cfg.table_cfg.prim_path, reset_xform_properties=False
        )

        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(color=(1.0, 1.0, 1.0))
        )

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        print("Scene is set up.")
~~~

#### pre-physics step calls
~~~py
    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self._process_action(actions)

    def _apply_action(self):
        # set left/right arm/gripper joint position targets
        self._robot.set_joint_position_target(
            self.left_arm_joint_pos_target, self.left_arm_joint_ids
        )
        self._robot.set_joint_position_target(
            self.left_gripper_joint_pos_target, self.left_gripper_joint_ids
        )
        self._robot.set_joint_position_target(
            self.right_arm_joint_pos_target, self.right_arm_joint_ids
        )
        self._robot.set_joint_position_target(
            self.right_gripper_joint_pos_target, self.right_gripper_joint_ids
        )
~~~

Before each physics step, `_pre_physics_step` processes the input actions to make them suitable for the current environmnet. `_apply_action` applies the processed actions to the robots.  

#### post-physics step calls  
**observation (key part):** 

~~~py
    # post-physics step calls
    def _get_observations(self) -> dict:
        # note: the position in observations should in the local frame

        # get robot end-effector (EE) pose
        left_ee_pos = (  # convert the position from the world position to the local cordinate system
            self._left_ee_frame.data.target_pos_w[..., 0, :] - self.scene.env_origins
        )
        right_ee_pos = (
            self._right_ee_frame.data.target_pos_w[..., 0, :] - self.scene.env_origins
        )
        left_ee_pose = torch.cat(
            [left_ee_pos, self._left_ee_frame.data.target_quat_w[..., 0, :]], dim=-1
        )
        right_ee_pose = torch.cat(
            [right_ee_pos, self._right_ee_frame.data.target_quat_w[..., 0, :]], dim=-1
        )

        joint_pos, joint_vel = self._process_joint_value()

        # get object pose
        object_pos = self._object[self.object_id].data.root_pos_w - self.scene.env_origins
        object_pose = torch.cat([object_pos, self._object[self.object_id].data.root_quat_w], dim=-1)

        obs = {
            # robot joint position: dim=(6+2)*2
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            # robot ee pose: dim=7*2
            "left_ee_pose": left_ee_pose,
            "right_ee_pose": right_ee_pose,
            # object pose: dim=7
            "object_pose": object_pose,
            # goal pose: dim=7
            "goal_pose": torch.cat([self.goal_pos, self.goal_rot], dim=-1),
            "last_joints": joint_pos,
        }
        if self.cfg.enable_camera:
            # image observations: N*(H*W*C)
            obs["front_rgb"] = self._front_camera.data.output["rgb"].clone()[..., :3]
            obs["front_depth"] = (
                self._front_camera.data.output["distance_to_image_plane"]
                .clone()
                .unsqueeze(-1)
            )
            obs["left_rgb"] = self._left_wrist_camera.data.output["rgb"].clone()[
                ..., :3
            ]
            obs["left_depth"] = (
                self._left_wrist_camera.data.output["distance_to_image_plane"]
                .clone()
                .unsqueeze(-1)
            )
            obs["right_rgb"] = self._right_wrist_camera.data.output["rgb"].clone()[
                ..., :3
            ]
            obs["right_depth"] = (
                self._right_wrist_camera.data.output["distance_to_image_plane"]
                .clone()
                .unsqueeze(-1)
            )

        return {"policy": obs}
~~~

**process andd organize joint position and velocity data:**
~~~py
    def _process_joint_value(self):
        joint_pos = self._robot.data.joint_pos.clone()
        ...

        joint_vel = self._robot.data.joint_vel.clone()
        l_arm_joint_vel = joint_vel[:, self.left_arm_joint_ids]
        r_arm_joint_vel = joint_vel[:, self.right_arm_joint_ids]
        l_gripper_joint_vel = joint_vel[:, self.left_gripper_joint_ids]
        r_gripper_joint_vel = joint_vel[:, self.right_gripper_joint_ids]
        # normalize gripper joint velocity
        l_gripper_joint_vel = l_gripper_joint_vel[:, 0] / (
            self.gripper_open - self.gripper_close
        )
        r_gripper_joint_vel = r_gripper_joint_vel[:, 0] / (
            self.gripper_open - self.gripper_close
        )

        joint_vel = torch.cat(
            [
                l_arm_joint_vel,
                l_gripper_joint_vel.view(-1, 1),
                r_arm_joint_vel,
                r_gripper_joint_vel.view(-1, 1),
            ],
            dim=-1,
        )
        return joint_pos, joint_vel
~~~

~~~py
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ...
    def _get_right_ee_pose(self):  # only used right arm (?)
        ...
~~~

**get rewards (key point):**
~~~py
    def _get_rewards(self) -> torch.Tensor:
        reward = self._compute_reward()
        return reward
~~~  

~~~py
    # reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
~~~

#### auxiliary methods