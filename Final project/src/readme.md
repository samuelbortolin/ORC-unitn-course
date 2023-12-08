# DQN implementation

## Folder structure:

- 03_assignment: this folder contains all the scripts needed to test the work we done. The 	main file si DQN. It also contains pretrained weights useful to test different trainings. The weights names are structured as follow: <number-of-joints_number-of-controls-discretizations>
- plots: this folder contains the cost to go history during training, the file names are structured as follow: <number-of-joints_number-of-controls-discretizations>
- videos: this folder contains the videos of the Greedy policy given by the network applied to the pendulum after training, the file names are structured as follow: <number-of-joints_number-of-controls-discretizations>
- tried_approaches: this folder contains different methods that we tried and then discarded due to bad performances. Priority_buffer must be tested in combination with SumTree.py
- utils: general robotics utils folder


## DQN file:

Inside the 03_assignment folder it is possible to launch the DQN.py file. It is possible to modify `nj` and `nu` variables in order to modify the number of joints and the discretization of the controls to use. In order to load the correct weights modify the `weights_file` variable and launch the file to visualize the greedy policy.

Instead, for launching the training set the flag `TRAINING` to True and wait until the completion. It is possible to interrupt it through a  `KeyboardInterrupt` and it will stop at the current iteration.
When lanuching the training is also possible to set `USE_EVALUATE_GREEDY_POLICY` to True in order to evaluate the Greedy policy given by the network and save the best weights.
