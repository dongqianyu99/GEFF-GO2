# Distilled Feature Fields Enable Few-Shot Language-Guided Manipulation

>*keywords:* **Distilled Feature Fields, Few-Shot, 2D-to-3D, Language-Guided Manipulation**

## Abstract

- 2D-to-3D for 3D geometry ==> accurate 3D geometry & 2D rich semantics
- *6-DOF grasping and placing* with few-shot learning method
- achieve in-the-wild generalization to unseen objects
- *features distilled fields* ==> vision-language model, *CLIP*

## Introduction

- given a *few grasping demonstrations or text descriptions* without having previously seen a similar item ==> pre-trained image embeddings

### workflow

*Step1:* scan by taking a sequence of photos ==> construct a *neural radiance field* **(NeRF)**