# Captain of Industry farm allocation optimisation

### Usage

Unfortunately, it's not very user friendly as this point.
To use this you need to:

0) Know how to build and run a command-line rust program.
1) Clone the repository
2) Edit number_of_farms_needed_for_each_crop in main.rs
3) Run in `--release` mode so it runs fast enough. 5-farm rotation is very slow to run since it calculates all possible crop rotations combinations. 6+ is faster as it randomly samples rotations. Random sampling is not guaranteed to give you the optimal solution, but it'll be pretty close and won't take hours/days/weeks to run.
