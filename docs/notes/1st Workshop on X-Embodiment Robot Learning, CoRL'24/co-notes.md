"Exploring the Boundaries of the X-Embodiment Problem" -- Ryan Julian (Google)  

What is X-embodiment?
- one model for many (all?) robots  
- the prompt for model is **Robot Agnostic**  
- the solution to the question (probably a vague request) is robot specific  

different from miximizing the probability:  
Multi-Task Policy Learning,  
$$
\mathbb{S} \times \mathbb{A} \times \mathbb{T} \times \mathbb{E}
$$

motion generalizationt is not yet appeared  
- trajectory distributions are kind of lumpy $\Rightarrow$ multimodal, most of those trajectory distributions are completely irrelevant    
$\Rightarrow$ not accessing a **continuum of robot embodiment**  

weak X-Embodiment  
- transfer between morphologically similar embodiments  
strong X-Embodiment  
- transfer information, training, policies, vqa, nevigation between any robot in the world to any others  

lack of Internet scale date set of embodiments  
state and action domains are very different, more importantly, the actual domains you can access with different robots are different $\Rightarrow$ can't transfer from one to another  
robots dynamics are different  

**Take the robots out of robotics**  

for Google deepmind:  
- start with specialist models    
- merge these models (brute force)  

一个单臂机械手和一个只用单臂的双臂机械手可以共享数据，可以做的事情是类似的，但做的方式是不同的
但单臂和双臂只能共享部分数据  

try to solve:  
use data representations which are embodiment agnostic  
$\Rightarrow$ arching trajectory, using a drawn trajectory representation to provide instruction and training data  
- a human to demonstrate what to do   
- the shape of the gripper doesn't matter  

如果数据是双臂的，那么单臂依然没法学会  
some tasks just can't be done by your robot $\Rightarrow$ using an embodied reasoning system to tell $\Rightarrow$ produce different output for a different embodiment  
the generated instructions and actions should all be different for different embodiment  

Some possible solutions  
- being flexible about data collection  
  - using generative data models, hypothetical videos of a human $\Rightarrow$ mapping from human to the robot  
- marginalize out that embodiment term not during pretraining but during post training $\Rightarrow$ fine tuning using a really small dataset, standard procedure  








