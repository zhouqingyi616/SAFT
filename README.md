# SAFT: shotgun advancing front technique
Code for paper ***"SAFT: shotgun advancing front technique for massively parallel mesh generation on graphics processing unit"***. Our paper has been accepted by ***International Journal for Numerical Methods in Engineering***. You can download it from [here](https://doi.org/10.1002/nme.7038).
  
***SAFT*** is a **front-based** parallel advancing front algorithm that runs on **single GPU**. 

Right now our implementation only includes **2D triangular mesh**. However by modifying the rules utilized when generating new elements, SAFT can be readily applied to non-triangular mesh / 3D mesh, so long as you're following the advancing front framework. 

This work has been done under the supervision of Prof. Zongfu Yu, as well as Prof. Qiqi Wang. Thanks a lot for their help! 

## Device
All results in the paper were obtained with my personal laptop, equipped with
- one NVIDIA Geforce RTX2060 graphics card
- one Intel Core i7-9750 CPU (@2.60 GHz)

## Requirements
All codes contained in this repository are written with CUDA C++.  

I'm using Visual Studio 2019, since I prefer working with Windows system.  

I've also tested compiling the source code using `nvcc` on Ubuntu, which also works well. 

Special requirements: 
- [CUDA](https://developer.nvidia.com/cuda-downloads): I've included CUDA 10.1 into "Build dependencies", but a slightly older version should also work. 
- [EasyX library](https://easyx.cn/), required if you want to visualize the generated mesh. 

## Results
Here are some of the results shown in the paper. Please refer to the paper for more details. 
### Example: violin
"Violin" case in paper (see section 3.1 for details). Please refer to our paper for a high-resolution image.   

(a) without Laplacian smoothing. (b) With Laplacian smoothing. 

![Violin](figures/violin.PNG?raw=true)

Note that the script used to visualize the above mesh is written in Python (relies on matplotlib), and is not included in this repository. 

### Speed & scalability
Using one single Geforce RTX2060 GPU, we are able to generate a 2D mesh containing **72.6M** elements in **less than 7 minutes**.  

For smaller cases, a generation speed of **233k elements per second** can be achieved with **3072 threads**.  

The following figure shows scalability: for given problem size, how the time consumption changes when increasing number of threads. Note that using too many threads does not necessarily lead to significant improvement (see section 3.3 for detailed discussion). 

![Scalability](figures/scale.PNG?raw=true)

## Files included
- **util_class.h**: defines important classes, including `vertex`, `face`, `QuadTree` and so on. 
- **shape_def.h**: defines the geometry as well as initial boundary of "ring" case. 
- **util_func_CUDA.h**: defines auxiliary functions that run on GPU. These lightweight functions will be called very often during mesh generation. 
- **util_func.h**: defines auxiliary functions (including visualization) that run on CPU. 
- **main.cu**: defines the `main()` function, as well as "find new elements" step and "calculate intersection" step. 

## How to modify
If you'd like to modify the code to mesh some arbitrary 2D domain, here are the steps:
- Turn the domain boundary into a vector of faces. Be aware of the normal vector direction! 
- Define the element size distribution by modifying `cuda_elem_size()` function.
- Choose the number of threads based on your problem size. As a starting point, it would be good to make sure the code runs correctly with a few threads. 

## Contact
In case you're interested in obtaining supplementary data or code not included here, you can contact me through e-mail: <qzhou75@wisc.edu>  

I'm always open to discussion, and will try my best to reply to received e-mails. 

## Cite
If you find the SAFT code provided here interesting and useful, please consider citing our paper:

    @article{zhou2022saft,
    title={SAFT: Shotgun advancing front technique for massively parallel mesh generation on graphics processing unit},
    author={Zhou, Qingyi and Wang, Qiqi and Yu, Zongfu},
    journal={International Journal for Numerical Methods in Engineering},
    volume={123},
    number={18},
    pages={4391--4406},
    year={2022},
    publisher={Wiley Online Library}
    }
