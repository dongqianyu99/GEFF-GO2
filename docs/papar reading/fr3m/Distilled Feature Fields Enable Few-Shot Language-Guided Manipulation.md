# Distilled Feature Fields Enable Few-Shot Language-Guided Manipulation

>*keywords:* **Distilled Feature Fields, Few-Shot, 2D-to-3D, Language-Guided Manipulation**

## Abstract

- 2D-to-3D for 3D geometry $\Rightarrow$ accurate 3D geometry & 2D rich semantics  
- *6-DOF grasping and placing* with few-shot learning method  
- achieve in-the-wild generalization to unseen objects  
- *features distilled fields* $\Rightarrow$ vision-language model, *CLIP*  

## 1. Introduction

- given a *few grasping demonstrations or text descriptions* without having previously seen a similar item $\Rightarrow$ pre-trained image embeddings  
- **DINO ViT**, *a self-supervised vision transformers* provides features (out-of-the-box visual descriptors for dense correspondence)  
- **CLIP**, *a vision-language model*, a strong <u>zero-shot</u> learner on various vision and visual question-answering tasks  
- rich visual and language priors within 2D foundation models $\Rightarrow$ *generalize to new categories of objects*  

### workflow

*Step1:* scan scene by taking a sequence of photos  
*Step2:* construct a *neural radiance field* **(NeRF)**, produce *Distilled Feature Field* **(DFF)**  
*Step3:* reference demonstrations and language instructions to grasp objects  

![alt text](image.png)

?>**NeRF**  
pre-trained vision foundation model (neural network) providing *image features*  
mutiple 2D images $\Rightarrow$ 3D scene  representation called *DFF*  
DFF embeds knowledge from 2D feature maps into a 3D volume  

### challenge

- *modeling time* $\Rightarrow$ *hierarchical hashgrids*  

?>**Hierarchical Hashgrids**  
Hierarchical hashgrids is a technique used to accelerate data querying and storage in 3D space.   
It organizes data using a *multi-level hash grid* structure, allowing for fast lookup and insertion operations.  

- *vision-language features:* *CLIP* produce image-level features, 3D feature distillation requires dense 2D descriptors $\Rightarrow$ *MaskCLIP* reparameterization trick, extracting dense patch-level features from CLIP   

## 2. Problem Formulation

>- a single rigid-body transformation is parameterized as ${T} \in {SE(3)}$  
>- parameterize a 6-DOP grasp or place pose as ${T} = {(R, t)}$, ${R}$ is the rotation matrix, ${t}$ is the ranslation vector  
>- given a set of RGB images $\{ {I} \}$ with corresponding camera poses  

### Few-Shot Manipulation

*learning:* each demonstration ${D}$ consists of the tuple $\langle \{ {I} \} , {T}^*\rangle$, ${T}^*$ is a pose that accomplishes the desired task  

*testing:* given multiple images $\{ {I}^\prime \}$ of a new scene which may contain distractor objects and clutter $\Rightarrow$ predict a pose ${T}$ that achieves the task  

>want to test for *open-ended generalization*: the new scene contains related but previously unseen objects that differ from the demo objects  

### Open-Text Language-Guided Manipulation

*testing:* provides the robot with a text query ${L}^+$ to specify which object to manipulate and negative texts ${L}^-$ to reject distractors  

>${L}^-$ can be sampled automatically (?)  

## 3. Feature Fields for Robotic Manipulation (F3RM)