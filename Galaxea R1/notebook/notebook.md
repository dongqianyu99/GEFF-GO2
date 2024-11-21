# Galaxea R1  

## Install Isaac Lab
Always error: **Failed to connect any GPU devices, including an attempt with compatibility mode.**  

ZSC solved the problem, concerning Linux core and NVIDIA drivers impropriate installation.  

installed: Isaac sim, cache, nucleus navigator, local Nucleus Service  

## Basic Operations  
[isaac sim具身智能仿真系列](https://www.bilibili.com/video/BV1TZ421g7xy?spm_id_from=333.788.videopod.sections&vd_source=14ad5ada89d0491ad8ab06103ead6ad6)  

First 10 classes are concerning basic operations in Isaac sim  

### Core API  
#### Hello World

baseline  
~~~py
rom omni.isaac.examples.base_sample import BaseSample

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

