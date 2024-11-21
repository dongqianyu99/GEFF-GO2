# Galaxea R1  

## Install Isaac Lab
Always error: **Failed to connect any GPU devices, including an attempt with compatibility mode.**  

ZSC solved the problem, concerning Linux core and NVIDIA drivers impropriate installation.  

installed: Isaac sim, cache, nucleus navigator, local Nucleus Service  

## Basic Operations  
### Standalone Python  
Isaac sim has a Python environment inside: ~/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh  
`./python.sh` there're verious Python APIs 

[isaac sim具身智能仿真系列](https://www.bilibili.com/video/BV1TZ421g7xy?spm_id_from=333.788.videopod.sections&vd_source=14ad5ada89d0491ad8ab06103ead6ad6)  

First 10 classes are concerning basic operations in Isaac sim  

### Core API  
#### Hello World

baseline  
~~~py
from omni.isaac.examples.base_sample import BaseSample

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):

        world = self.get_world()
        world.scene.add_default_ground_plane()
        return

    async def setup_post_load(self):
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return

~~~

add a cube  
~~~py
from omni.isaac.core import World
import numpy as np
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.examples.base_sample import BaseSample

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


class HelloWorld(BaseSample):
    ...

    def setup_scene(self):

        # world = self.get_world()
        world = World.instance()  # retrieve the current instance of the World across files and extension
        world.scene.add_default_ground_plane()

        fancy_cube = world.scene.add(
            DynamicCuboid(
                prim_path = "/World/random_cube",
                name = "fancy_cube",
                position = np.array([0, 0, 1.0]),
                scale = np.array([0.5015, 0.5015, 0.5015]),
                color = np.array([1, 0, 0]),
            )
        )

        return

    ...
~~~

the things done after **Load**  
~~~py
async def setup_post_load(self):
    world = World.instance()
    self._cube = world.scene.get_object("fancy_cube")
    position, orientation = self._cube.get_world_pose()
    linear_velocity = self._cube.get_linear_velocity()

    print("Cube position is : " + str(position))
    print("Cube's orientation is : " + str(orientation))
    print("Cube's linear velocity is :" + str(linear_velocity))

    return
~~~

**call back function**  
~~~py
async def setup_post_load(self):
    world = World.instance()
    self._cube = world.scene.get_object("fancy_cube")
    self._world.add_physics_callback("sim_step", callback_fn = self.print_cube_info)  # will be called before each physics step

    return

def print_cube_info(self, step_size):
    position, orientation = self._cube.get_world_pose()
    linear_velocity = self._cube.get_linear_velocity()

    print("Cube position is : " + str(position))
    print("Cube's orientation is : " + str(orientation))
    print("Cube's linear velocity is :" + str(linear_velocity))
~~~

## 