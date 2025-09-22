# Captain of Industry farm allocation optimisation

### Usage

Unfortunately, it's not very user friendly as this point.
To use this you need to:

1) Close the repositort
2) Edit number_of_farms_needed_for_each_crop in main.rs
3) Run in `--release` mode so it runs fast enough. 5-farm rotation is very slow to run since it calculate all possible rotations. 6+ is faster as it randomly samples rotations.
