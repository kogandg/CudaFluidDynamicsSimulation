# CudaFluidDynamicsSimulation

A fluid dynamics simulation in C++ using CUDA to simulate and SFML to display

Based on [GPU Gems 1, Chapter 38](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)

# Introduction 

The goal of this was to learn CUDA and create a fun simulation to have a visual product of the learning. 

Fluid simulation is a solid base block for simulating lots of interesting phenomena like smoke curling, steam from tea, or paint mixing. This problem is also particularly suited to CUDA GPU programming as it is very parallelizable. 

# The Navier-Stokes Equation for Incompressible Flow

The Navier-Stokes equation can find the state of a fluid over time given a velocity and pressure are known for some initial time. In general, it is a problem and remains unsolved, however one can find analytical solutions to the equation, which is the goal of this program. First, let's understand the formula itself and its terms. 

The Navier-Stokes Equation

![image](https://github.com/user-attachments/assets/fc21c07a-b330-488f-9db0-9567c5237a3f)

The equation is divided into 5 parts:

- ![image](https://github.com/user-attachments/assets/b76cd523-d8d3-40fc-9907-bd1e798238aa) - Velocity:
 The rate of change in the velocity of the fluid, the goal of this simulation is to calculate this value for each particle. 
- ![image](https://github.com/user-attachments/assets/4a515d1e-cccf-4d02-8d0b-470cf8b552dc) - Advection:
The velocity of a fluid causes the fluid to transport properties from one location to the next such as other objects, densities, and other quantities along with the flow. This term represents self-advection of the velocity field, meaning that the fluid carries itself along the field.
- ![image](https://github.com/user-attachments/assets/57458a35-f15f-4d0d-9f45-3b81316b24d1) - Pressure:
This is the pressure that is exerted on each particle. When a force is applied to a fluid, it doesn't instantly propagate through the fluid, but rather particles close to the force push on the particles further away, and the pressure builds up. 
- ![image](https://github.com/user-attachments/assets/8bf14296-79d8-43bb-8889-3a01f02bf682) - Diffusion:
This is the viscosity of the medium, so the larger it is, the stronger and more the fluid resists the force that is applied to it. This resistance results in a diffusion of the momentum, and thus the velocity. 
- ![image](https://github.com/user-attachments/assets/ad936cd9-c863-4f8d-b87d-4d6ce1d17ab5) - External forces:
This last term is the external forces that we apply to the liquid. Later in this case the external force is applied by the user. 

Since this fluid is incompressible, there is another equation:
![image](https://github.com/user-attachments/assets/492fbdd7-f218-4902-81fe-2d6d6b9fe317)
Which simply states that the energy in the medium is constant. 

# Quick mathematical background review

As was shown above with the overview of the Navier Stokes equation, the understanding of the equation is very important to finding numerical solutions, and requires a mathematical understanding of the process. 

First, we want to mathematically represent the state of the fluid. To define the state of the fluid we need to represent its velocity, the velocity determines how the fluid moves itself, and properties and objects that are in it. This velocity varies in both time and space, so we will represent it as a vector field. 

A vector field is a mapping of a vector-valued function onto some parameterized space, in this case we will use a Cartesian grid, however other parameterizations are possible. So, the velocity vector field of our fluid is defined so that every position **x** = (x, y) has an associated velocity at some time t, **u**(**x**, t). This field will look like:

![image](https://github.com/user-attachments/assets/a412b793-8bfd-4a18-98b9-2b4a10f60401)


# Vector calculus review
![image](https://github.com/user-attachments/assets/d26ab5f5-cd0a-42f6-a315-10639ea241b8)

The operators above are used in the Navier Stokes equation, and all have different uses and meanings. 

The gradient operator of a scalar field is a vector of the parietal derivatives of that scalar field. 

Divergence has a very important physical property, it is the rate at which “density” leaves some region of space. In the Navier-Stokes equation, it applies to the velocity, so it measures the net change in velocity in a small region of the fluid. It is also used in the second equation which ensures that energy is conserved, by enforcing that the fluid is incompressible, by ensuring that the fluid has no divergence. The dot product in the divergence means that it results in a sum of the partial derivatives, so it returns a scalar, and can only be applied to a vector field.

Last is the Laplacian operator, which is a combination of the gradient and the divergence operators. We see that the gradient is applied to a scalar field, and produces a vector field, and the divergence is applied to a vector field, and produces a scalar field. The Laplacian is the result of applying the gradient operator, then applying the divergence operator. 

A simplification we can make is so assume that ![image](https://github.com/user-attachments/assets/4879f758-472b-42a0-acd8-447c88f5f2f9). This does not greatly affect the accuracy of the algorithm, but it does reduce the operations per iterations, and also makes the expressions easier to read and understand. 

# Solving the equations

The Navier-Stokes equations can be solved analytically for a few very simple configurations, but using numerical integration techniques, the equations can be solved incrementally. In this case where a fluid simulation, and seeing the fluid and flow change over time, an incremental numerical solution fits perfectly. 

# Particle Handling and Advection

To solve for any of the terms of the equation, we need to update the velocity at each particle. Because we are computing how some quantity moves along a velocity field, it is helpful to image a particle as a grid cell.  

To calculate advection, the first thought might be to update the grid as we would a particle system, moving the position, r, of each particle along the velocity field by the distance it would travel in time t. 

![image](https://github.com/user-attachments/assets/87346e3b-d154-4737-bb0f-af3373a7bf4b)

This might be recognized as Euler's method, a simple and explicit/forward integration method for ordinary differential equations. 

However, there are two main problems with using this method, the first is that simulations that use explicit methods for advection are unstable when using large time steps; they can “blow up” if the magnitude of u(t)t is larger than the size of a single grid cell. The other problem is actually implementing this on the GPU, since we implement it in fragment programs which can’t change the locations of the fragments they are writing, which can’t be done as the forward integration requires us to “move” the particles.
