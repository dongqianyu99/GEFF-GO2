# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis  

>*keywords:* view synthesis, image-based rendering, volume rendering, 3D deep learning  

## 1. Introduction  

***Problem:*** view synthesis  

- represent a static scene as a continuous 5D function $\Rightarrow$ $(x, y, z)\;(\theta, \phi)$  
- outputs the **radiance** emmitted in each point  
- a **density** at each point $\Rightarrow$ differential opacity controlling how much radiance is accumulated by a ray passing through $(x, y, z)$  
- MLP  
- transforming input 5D coordinates with a **positional encoding**  
- **hierarchical sampling procedure**  
- novel views  

![alt text](image.png)  

?>**MLP**  
*MLP (Multilayer Perceptron)* is a type of feedforward neural network that typically consists of multiple layers, including an **input layer**, **hidden layers**, and an **output layer**. Each layer is composed of multiple neurons (nodes) connected to each other through weighted connections.

## 2. Related Work  

## 3. Neural Radiance Field Scene Representation  