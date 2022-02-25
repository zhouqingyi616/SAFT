# SAFT: shotgun advancing front technique
Code for paper ***"SAFT: shotgun advancing front technique for massively parallel mesh generation on GPU"***.  
  
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
Using one single Geforce RTX2060 GPU, we are able to generate a 2D mesh containing 72.6M elements in less than 7 minutes.  

For smaller cases, a generation speed of 233k elements per second can be achieved with 3072 threads.  

The following figure shows scalability: for given problem size, how the time consumption changes when increasing number of threads. Note that using too many threads does not necessarily lead to significant improvement (see section 3.3 for detailed discussion). 

![Scalability](figures/scale.PNG?raw=true)

## Files included

## How to modify

## Contact

## Cite
    @article{zhou2022saft,
      title={SAFT: shotgun advancing front technique for massively parallel mesh generation on GPU},
      author={Zhou, Qingyi and Wang, Qiqi and Yu, Zongfu},
      journal={XXX},
      year={2022}
    }
  
