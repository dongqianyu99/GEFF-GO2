# 3D Gaussian Splatting for Real-Time Radiance Field Rendering  

>key words: **high-quality**, **real-time**, **novel-view**, **unbounded and complete scenes**  

[b站人话讲解](https://www.bilibili.com/video/BV1zi421v7Dr/?spm_id_from=333.337.search-card.all.click&vd_source=14ad5ada89d0491ad8ab06103ead6ad6)

***Tree Key Elements:***  
- starting from *sparse points produced during camera calibration*, we represent the scene with **3D Gaussians** that preserve desirable properties of *continuous volumetric radiance fields* for scene optimization while avoiding unnecessary computation in empty space  

>**Camera Calibration:**  
This process involves determining the intrinsic and extrinsic parameters of the camera, which allows for accurate understanding of the relationship between the camera and objects in the scene. *Sparse points generated* during calibration are often used as the basis for scene reconstruction.  

>**3D Gaussians:**  
A mathematical model used to represent *uncertainty or distribution* of points or regions in 3D space. By using 3D Gaussians, each point in the scene **not only has a spatial coordinate but also represents a potential distribution** (such as volume density), which is highly effective for modeling continuous volumetric radiance fields.  

>**Continuous Volumetric Radiance Fields:**  
This refers to the **volume density** and **radiance** at every point in the scene, which can be estimated through volume rendering techniques. The use of 3D Gaussians enables the scene to *maintain these continuous radiance properties* while allowing for scene optimization.

- perform **interleaved optimization/density control of the 3D Gaussians**, notably optimizing **anisotropic convariance** to achieve an accurate representation of the scene  

>**Interleaved Optimization/Density Control:**  
This refers to the alternating optimization of different parameters. In this context, it involves optimizing both the density (i.e., the weight and spread of each 3D Gaussian distribution) and the associated covariance structure.  

>**Anisotropic Covariance:**  
The covariance matrix describes the relationship between random variables. Anisotropy means that the *covariance may differ in different directions* (as opposed to isotropy, where covariance is the same in all directions). Optimizing anisotropic covariance in 3D Gaussians aims to more accurately represent the geometry and details of the scene, capturing directionally varying features in the real world.

-  a fast **visibility-aware rendering** algorithm that supports **anisotropic splatting** and both accelerates trainning and allows real-time rendering  

>**Visibility-aware Rendering Algorithm:**  
This means that the rendering algorithm takes into account *which parts of the scene are visible* (i.e., observable), and allocates computational resources more efficiently by *rendering only the visible parts*. This reduces unnecessary computation and speeds up the rendering process.  

>**Anisotropic Splatting:**  
**Splatting** is a rendering technique often used to spread image or geometric information *from discrete sample points to a continuous space*. Anisotropic splatting refers to the process where the influence of sample points is spread differently in different directions, meaning **the rendering effect adjusts based on the orientation** (or other geometric information) of the sample points. This method is more flexible than traditional isotropic splatting and can better capture directional geometric features in a scene.  

## 1. INTRODUCTION  

- *meshes* and *points* $\Rightarrow$ explicit, good fit for fast GPU/CUDA-based rasterization  
- NeRF $\Rightarrow$ optimizing a Multi-Layer Perceptron (MLP) using *volumeric* ray-marching  
- *interpolaing values* is the most efficient radiance field solutions to date build on continuous representations  
$\Rightarrow$ stochastic sampling is costly and can result in noise  

**tile-base splatting solution**  

**Three main components:**  
- initialize the set of 3D Gaussions with the **sparse point cloud** produced for free as part of the SfM (Structure-from-Motion) process $\Rightarrow$ only SfM points as input or even *random initialization*  

>**Structure-from-Motion (SfM):**  
SfM is a computer vision technique aimed at reconstructing the 3D structure of a scene and estimating the positions and orientations of the cameras from a series of 2D images. By analyzing the relative motion between images, SfM can reconstruct the 3D geometry of the scene from multiple viewpoints.  
The initial reconstruction in SfM is usually **sparse**, meaning it uses a small number of feature points to reconstruct the scene.  

- optimization of the properties of the 3D Gaussians - **3D position, opacity $\alpha$, anisotropic covariance, and spherical harmonic (SH) coefficients** $\Rightarrow$ *compact, unstructured, and precise representation of the scene*    
- real-time rendering solution (fast GPU sorting algorithms, tile-based rasterization)  
  - 3D Gaussian representation $\Rightarrow$ perform **anisotropic splatting** that *respects visibility ordering* (sorting, $\alpha$ blending)  
  - a fast and accurate *backward pass* by tracking the traversal of as many sorted splats as required  

![alt text](v2-7cbe3b0c3b67ce80593fad0d73a814b5_r.png)  

## 2. RELATED WORK  

### Traditional Scene Reconstruction and Rendering  
- based on light Fields  
- Structure-from-Motion (SfM)  
- multi-view stero (MVS)  
- neural rendering algorithm  

### Neural Rendering and Radiance Fields  
- NeRF $\Rightarrow$ Mip-NeRF360  

### Point-Based Rendering and Radiance Fields  
- Point-based methods render disconnected and unstructured geometry samples $\Rightarrow$ **splatting** point primitives with an extent larger than a pixel, e.g., *circular or elliptic discs, ellipsoids, or surfels*  
- *differentiable* point-based rendering techniques $\Rightarrow$ Points be augmented with neural features and rendered using a CNN  
- **Point-based** *$\alpha$-blending* and *NeRF-style* volumetric rendering $\Rightarrow$ color C is given by volumetric rendering alog a ray  
>the rendering algorithm is very different: **NeRF** vs **Point-based**  

*NeRF:*  
- continuous representation implicitly representing empty/occupied space  
- expensive random sampling with noise and computational expense  

*Point-based:*
- unstructured, discrete representation  
- allow creation, destruction, and displacement of geometry $\Leftarrow$ optimizing opacity and positions  
- avoiding the shortcomings of a full volumetric representaion  

- *fast sphere rasterization* $\Rightarrow$ **tile-based and sorting renderer**  
- "Our rasterization respects **visibility order**"  
- "We back-propagate gradients on all splats in a pixel and rasterize anisotropic splats"

## 3. OVERVIEW  

a set of images of a static scene + corresponding cameras calibrated by SfM $\Rightarrow$ sparse point cloud $\Rightarrow$ 3D Gaussians (**position**, **covariance matrix**, **opacity $\alpha$**)   
$\Rightarrow$ compact representation of the 3D scene (by **highly anisotropic volumetric splats**) + color of the radiance field (by **spherical harmonics, SH**)  

Why 3D Gaussian is fast?  
**tile-based rasterizer** $\Rightarrow$ *$\alpha$ blending of anisotropic splats*, *respecting visibility order*, *fast backward pass*  

## 4. DIFFERENTIABLE 3D GAUSSIAN SPLATTING  

### Modeling
model the geometry as a set of ***3D Gaussians*** $\Rightarrow$ **do not require normals**:  

- 3D covariance matrix $\Sigma$  
- in world space centered at point (mean) $\mu$  
$$
G(x)=e^{-\frac{1}{2}(x)^T \Sigma ^{-1} (x)}
$$  

### Projection  
3D Gaussians to 2D for rendering  
</br>
Given viewing transformation $W$, the covariance matrix $\Sigma^{\prime}$ in camera coordinate:  
$$
\Sigma^{\prime} = JW \Sigma W^T J^T
$$
- $J$ is the *Jacobian* of the affine approximation of the projective transformation  

### Optimization  
!>directly optimize the covariance matrix $\Sigma$ $\Rightarrow$ covariance matrices have *physical meaning* only when they are opsitive **semi-definite** $\Rightarrow$ gradient descent cannot be easily constrained, matrices might be invalid  

$\Sigma$ is analogous to describing the configuration of an **ellipsoid**  

Given a **scaling matrix $S$** and **rotation matrix $R$**:  
$$
\Sigma = RSS^T R^T
$$  

for *independent optimization*, store them separatly: 3D vector $s$ for scaling, quaternion $q$ for rotation  
- derive the gradient for all parameters explicitly  

## 5. OPTIMIZATION WITH ADAPTIVE DENSITY CONTROL OF 3D GAUSSIAN  

optimization is the core  

optimize position $p$, $\alpha$, covariance $\Sigma$, **SH coefficients representing color $c$ of each Gaussian** (interleaved with steps that control the *density*)  

### 5.1 Optimization  

incorrectly positioned due to 3D to 2D $\Rightarrow$ *destroy* or *move* geonetry  

**quality** of the parameters of the covariances of the 3D Gaussians is critical $\Rightarrow$ *large homogeneous areas* can be captured with a samll number of large anisotropic Gaussians     

**Stochastic Gradient Descent** + *fast rasterization* $\Rightarrow$ WHY SO FAST

$\alpha \Rightarrow$ **sigmoid activation function** $\Rightarrow [0, 1)$ range and obtain smooth gradients  
the scale of the covariance $\Rightarrow$ **exponential activation function**  

- estimate the initial covariance matrix as an *isotropic Gaussian*  
- axes equal to the mean of the distance to the closest three points  
- **standard exponential decay scheduling technique**, with loss function:  
$$
\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{D-SSIM}
$$

### 5.2 Adaptive Control of Gaussians  








