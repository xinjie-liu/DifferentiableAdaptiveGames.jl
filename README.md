# DifferentiableAdaptiveGames.jl

[![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

An adaptive game-theoretic planner that jointly infers players' objectives and solves for generalized Nash equilibrium trajectories, enabled by differentiating through a trajectory game solver. The solver is made end-to-end differentiable and supports direct combination with other learning-based components. This is a software package produced by research [Learning to Play Trajectory Games Against Opponents with Unknown Objectives](https://xinjie-liu.github.io/projects/game/). Please consult our project website for more information. 

```
@article{liu2023learning,
  author={Liu, Xinjie and Peters, Lasse and Alonso-Mora, Javier},
  journal={IEEE Robotics and Automation Letters}, 
  title={Learning to Play Trajectory Games Against Opponents With Unknown Objectives}, 
  year={2023},
  volume={8},
  number={7},
  pages={4139-4146},
  doi={10.1109/LRA.2023.3280809}}
```

<a href ="https://arxiv.org/abs/2211.13779"><img src="https://xinjie-liu.github.io/assets/img/liu2023ral_teaser.png"></a>

<a href ="https://xinjie-liu.github.io/assets/pdf/Liu2023learningPoster(full).pdf"><img src="https://xinjie-liu.github.io/assets/img/liu2023ral_poster.png" width = "560" height = "396"></a>



## Content

This package contains a nonlinear trajectory game solver (`/src/`) based on a mixed complementarity problem (MCP) formulation. The solver is made differentiable by applying the implicit function theorem (IFT). Experiment code employing this differentiability feature for adaptive model-predictive game-play is provided in `/experiment/`. Folder `/experiment/DrivingExample/` contains code for the ramp-merging experiment in our paper, and hardware experiment code on Jackal UGV robots is in `/experiment/Jackal/`. 

Acknowledgments: Most of the infrastructure used in the Jackal experiment is from a past project by [@schmidma](https://github.com/schmidma), who provided a package for controlling the Jackal robot in ROS using Julia code. 

Note: This repository contains the original solver implementation used in our research, "Learning to Play Trajectory Games Against Opponents with Unknown Objectives." Besides, we also published a more optimized implementation for the differentiable game solver ([link](https://github.com/JuliaGameTheoreticPlanning/MCPTrajectoryGameSolver.jl)). We kindly ask you to cite our paper if you use either of the implementations in your research. Thanks!

## Run a Four-Player Collision Avoidance Example

1.  Open REPL (Julia 1.9) in the root directory: `julia`

2. Activate the package environment by hitting `]` to enter the package mode first and then type: `activate .`

3. Instantiate the environment in the package mode if you haven't done so before by typing `instantiate`

4. Exit the package mode by hitting the backspace key; precompile the package: `using MCPGameSolver`

5. Run the example: `MCPGameSolver.main()`

## Run the Ramp Merging Experiment

1. Open REPL (Julia 1.9) in the directory `/experiment/DrivingExample`: `julia`

2. Activate the package environment by hitting `]` to enter the package mode first and then type: `activate .`

3. Instantiate the environment in the package mode if you haven't done so before by typing `instantiate`

4. Exit the package mode by hitting the backspace key; precompile the package: `using DrivingExample`

5. Run the ramp merging example: `DrivingExample.ramp_merging_inference()`

## Run the Two-Player Tracking (Jackal) Experiment

1. Open REPL (Julia 1.9) in the directory `/experiment/Jackal`: `julia`

2. Activate the package environment by hitting `]` to enter the package mode first and then type: `activate .`

3. Instantiate the environment in the package mode if you haven't done so before by typing `instantiate`

4. Exit the package mode by hitting the backspace key; precompile the package: `using Jackal`

5. Run the tracking example: `Jackal.launch()`

## License

This package uses PATH solver (via [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl)) under the hood. Larger-sized problems require to have a license key. By courtesy of Steven Dirkse, Michael Ferris, and Tudd Munson, temporary license keys are available free of charge. For more details about the license key, please consult [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl) (License section). Note that when no license is loaded, PATH does not report an informative error and instead may just report a wrong result. Thus, please make sure that the license is loaded correctly before using the solver.
