# DifferentiableAdaptiveGames.jl

[![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

An adaptive game-theoretic planner that jointly infers players' objectives and solves for generalized Nash equilibrium trajectories, enabled by differentiating through a trajectory game solver. This is a software package produced by research [Learning to Play Trajectory Games Against Opponents with Unknown Objectives](https://xinjie-liu.github.io/projects/game/). Please consult the project website for more information. 

<a href ="https://arxiv.org/abs/2211.13779"><img src="https://xinjie-liu.github.io/assets/img/liu2023ral_teaser.png"></a>

```
@article{liu2022learning,
  title={Learning to Play Trajectory Games Against Opponents with Unknown Objectives},
  author={Liu, Xinjie and Peters, Lasse and Alonso-Mora, Javier},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2023}
}
```
## Run the Ramp Merging Experiment

1. Open REPL (Julia 1.9) in the directory `/experiment/DrivingExample`

`julia`

2. Activate the package environment

`]activate .`

3. Instantiate the environment if you haven't done so before

`]instantiate`

4. Precompile

`using DrivingExample`

5. Run the ramp merging example

`DrivingExample.ramp_merging_inference()`

