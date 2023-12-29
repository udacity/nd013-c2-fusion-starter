# Writeup: ID_S1_EX2 3D lidar data inspection

In the images below we can see a 3d representation from the Lidar data from a sequence of the Waymo Open Dataset. 

## Vehicles with varying degrees of visibility
All 6 of these frames have been taken at different times. This can be seen well by the fact that there is a car overtaking the vehicle in the frames 3-6. This car is not visible in the front view of the car until it overtakes us and then it becomes visible in the front view.

Images 2 and 6 show the same 4 cars but at different points in time. In the picture 2 they are closer and therefore we can distinguish the individual features on the vehicles better than in the 6th picture where they are all further away.

## Vehicle features
In the frames 1, 3 and 4 I highlighted the rear windows of the car which can again be seen in all of these frames but from a different distance to that car.

In the frame 5 we can very nicely distinguish the wheels of the car next to us. These are however less stable over time than the rear features of the car as they are only visible during overtake manoeuvres.

![Lidar images overview](img/lidar_overview.PNG "Overview of six Lidar images with different PoVs")
