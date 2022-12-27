# New Task: Cube at Different Location
For this task, I changed the starting configuration of the cube to be x = 0.75,
y = -0.6, phi = 0.0. I opted to use the same feedforward + PI controller that 
is used in the *best* case, which has controller gains Kp = 2.0 and Ki = 0.03.

As with the other two cases, the error plots have a spike, this time at the start. In 
these plots however, the spike at the start of the error is the only irregularity and
the remainder of the plots look as expected, since they have a relatively smooth 
response which converges to near zero error.
