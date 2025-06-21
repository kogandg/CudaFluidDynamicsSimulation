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

The solution is to flip the problem and use an implicit method where instead of advecting quantities by where the particle moves over the current time step, we trace back each particle from each grid cell back in time to its previous position, and copy the quantities at the position to the starting cell. 

![image](https://github.com/user-attachments/assets/de10c411-f30b-4b15-94dd-c26dcfd92954)


Then to update the quantity q, we use:
![image](https://github.com/user-attachments/assets/21d5a6bc-a610-48ab-b7aa-3deafad114af)

It's important to note that at a low  velocity, we may not go out of the grid cell, so we should choose a correct minimum momentum force that the user can put onto the particles. 

Also to avoid loss of accuracy, in the case that the projection hits the boundary of a cell or its a non-integer value, we perform a bilinear interpolation of the values of the 4 nearest particles and use that as the true value at that point. Then this value is written to the starting grid cell. 

# Diffusion and Viscosity

Every fluid has some viscosity, which prevents external forces affecting it self, like honey or water. Viscosity directly affects the acceleration quired by a fluid:

![image](https://github.com/user-attachments/assets/329718ce-81e4-48ca-9331-8ad40b4642b0)

Like advection, we can formulate the problem simply with:

![image](https://github.com/user-attachments/assets/7c8bd5b2-2dc2-48d1-8682-233f99c9d8f8)

Using Euler's method, but again it is unstable for large time steps and velocities, so instead an implicit formula is used which looks like:

![image](https://github.com/user-attachments/assets/a45dcc12-50f1-4617-9749-a97e8d37967c)

Where I is the identity matrix. This form is stable for any timesteps and velocity. It can be solved using a Jacobi method discussed in the next sections

# Pressure

Pressure is what prevents particles from filling all the space available after some external force is applied to them. It is very difficult to calculate in the form it is represented in the Navier-Stokes equation, but is greatly simplified by applying Helmholtz’s decomposition theorem:  

![image](https://github.com/user-attachments/assets/39c74927-33bc-421f-bcba-102492c059ac)

This theorem states that W, a vector field, can be represented by a sum for two other vector fields. We say that W is the field obtained by calculating displacement, external forces, and diffusion. It has a non-zero divergence, which is contradicting the incompressibility condition of our fluid. To correct this, the pressure must be calculated, so u is the field with zero divergence. By applying the divergence operator we get the formula for the scalar pressure field: 

![image](https://github.com/user-attachments/assets/bc2b2436-7fcf-47a7-b1c4-c01f3d89ea55)

This expression is the Poisson equation for pressure, which can also be solved using a Jacobi method mentioned above. 

# Solving Poisson equations with the Jacobi method

We have two Poisson equations, one for pressure and the other for diffusion. They can be solved using an interactive method with the iterative equation:

![image](https://github.com/user-attachments/assets/b88a0876-3501-4b9c-9189-7bdd99c02656)

In our case, x is the element of an array which represents the scalar or vector field, k is the iteration number. K can be changed to either increase the accuracy of the calculation, or reduce it to increase the speed. 

To calculate the diffusion,![image](https://github.com/user-attachments/assets/3081535a-88e1-49eb-a3fe-74c2b0734107), and ![image](https://github.com/user-attachments/assets/91497473-a6e9-4312-847e-4afd1e5fbc3f), ![image](https://github.com/user-attachments/assets/3a4c0254-f68b-47be-a6bd-b17a4c613a2f). Here beta is the sum of the weights. To have this run on the GPU, we must store at least two velocity vector fields, to independently read values from one field, and write values to the other. It takes about 20-50 iterations to calculate the velocity field using this Jacobi method, which is quite a lot if it were performed on the CPU, but is not a problem when on the GPU. 

To calculate pressure, we say that ![image](https://github.com/user-attachments/assets/861ae2d1-2d1d-4415-a75e-704c166490a4), ![image](https://github.com/user-attachments/assets/8f1a2c62-8df1-4fdd-a599-3d62d8304234), ![image](https://github.com/user-attachments/assets/a31c90b7-4542-4090-bd96-eb74116c6cd4), and ![image](https://github.com/user-attachments/assets/135c9ae9-7452-4d9a-82b8-3a8383954597). The result is a value ![image](https://github.com/user-attachments/assets/a171112f-a37d-401e-bd30-6046fc4a89ea) at that point. Since we only use this to calculate the gradient subtracted from the velocity field, we don't need to do any more conversions. For a pressure field, it takes about 40-80 iterations, as with smaller values, the inaccuracies become quite noticeable. 

# External Forces

This is the simplest part of the algorithm. We describe the external forces by:

![image](https://github.com/user-attachments/assets/ebf6cc6c-bce6-4c1e-885f-c31242d5710b)

Where x_p, y_p are the mouse position, x, y is the position of the current cell, and r is the radius/scaling parameter. The momentum vector G is simply the difference between the original mouse position and the current mouse position. 

# Initial and Boundary Conditions

All differential equations in a finite domain need boundary conditions to be well defined. In this case, the boundary conditions are set to control the fluid near the edges of the coordinate grid and the initial conditions set the parameters that the particles have at the beginning of the simulation. 

In this case, the initial conditions are very simple. The fluid will be stationary, meaning zero velocity in any particle, and the pressure will also be zero. 

Boundary conditions are defined such that the velocity applied to the particles at the edges will be opposite to their current velocity, so they will be repealed from the edge, and so that the pressure is equal to the value right next to the boundary. These will be applied to all the bounding elements, seen below:

![image](https://github.com/user-attachments/assets/dd5d5d35-bbed-4494-b2c3-8219d66b50f8)
