# Implementation of turbulent models in NVIDIA modulus

## K-Epsilon model
K-eps model implementation expects 3D geometry.
Initialization of the flow with basic constants - nu, y_start
Re is calculated from nu
Two options are available - transient and steady state version
Time window size is set according to your needs and geometry size, default is 10s (transient version only)
Number of time windows determines how many neural nets will be trained sequentially - default is 1, but for better results it should be 5 to 10. 
Geometry is generated dynamically - solid body is loaded and wind tunnel is created through tesselation
Inlet - for now it is set to custom rectangular, can be adapted to tanh-driven inlet from ahmed body example
NNets - fully connected for u_tau and fourier nets for rest of the nets, they tend to converge better. u/v/w part can be also done with fully connected.
### Boundary conditions :
    - inlet - u/v/w set according the function
    - outlet - p set to 0
    - tunnel walls - simple no slip boundary (0 flow)
    - solid body - no slip boundary + velocity wall normal and parallel set to 0. Shear stress and ep not 0!
    - IC for interior traditionally 0 momentum/continuity and k+ep
    - flow init at t=0 zero, also k, ep and p
    - interior init, 0 flow and pressure
    - IC for time window - zero diff for prev step for flow (maybe should be pressure too, havent tried yet, TODO)
