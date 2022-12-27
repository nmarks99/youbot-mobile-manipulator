# ME 449 Final Capstone Project
This program generates a trajectory for the youBot robot simulated in CoppelliaSim. 
The code works by first generating a reference trajectory with the `TrajectoryGenerator`
function.

Once a reference trajectory has been created, the program enters a loop
which does the following:
- Calls `FeedbackControl()` which runs a feedforward + PI controller based
on the current configuration and reference configuration to obtain control velocities
to keep the robot on the reference trajectory.
- Call `NextPlan()` which uses the control velocities from the feedback controller to 
compute the configuration of the robot at the next step with a first order Euler step
- Call `Tse_from_config()` to find the configuration of the end effector with forward kinematics.
- Save the robot configuration and error in an array

After the main program loop completes, write the robot configuration and error history arrays to
CSV files and plot the error data. The error data plot will also be saved to a PNG file.

# Usage
To run the program, simply navigate to the "code/" directory in a terminal 
and run the `main.py` script with the appropriate argument:

- To run the *best* case, run the following command:
`python3 main.py best`
- To run the *overshoot* case, run the following command
`python3 main.py overshoot`
- To run the *newTask* case, run the following command
`python3 main.py newTask`


