# Writeup: Track 3D-Objects Over Time

Please use this starter template to answer the following questions:

### 1. Write a short recap of the four tracking steps and what you implemented there (filter, track management, association, camera fusion). Which results did you achieve? Which part of the project was most difficult for you to complete, and why?

For this midterm project in the Sensor Fusion course, the Waymo Open Dataset was used. Sensor data and calibration parameters were extracted from the dataset, and a LiDAR point cloud was constructed from the provided range images and visualized (ID_S1_EX2). The point cloud was then transformed into a Bird’s-Eye View (BEV) representation with ROI filtering/discretization and feature maps for intensity and height (ID_S2_EX1, ID_S2_EX2, ID_S2_EX3).

To understand the 3D object detection workflow, the inference pipeline of the [SFA3D model](https://github.com/MooyoungLee/SFA3D) was reviewed, covering the full process of LiDAR → BEV → Model → Decode → Post-processing → Visualization → Evaluation.

Using the transformed BEV data—including LiDAR intensity, height, point density, and spatial (x, y) coordinates—both FPN-ResNet and Darknet-based models were applied for object detection (ID_S3_EX1-3, ID_S3_EX1-4, ID_S3_EX1-5), and detections were converted into 3D bounding boxes (ID_S3_EX2). Detection performance metrics (IoU/center deviations and precision/recall per frame and in aggregate) were implemented and evaluated (ID_S4_EX1, ID_S4_EX2, ID_S4_EX3), and the results were visualized across 100 data frames.

### 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)? 


### 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?

#### Find and display 10 examples of vehicles with varying degrees of visibility in the point-cloud

#### Identify vehicle features that appear as a stable feature on most vehicles (e.g. rear-bumper, tail-lights) and describe them briefly. Also, use the range image viewer from the last example to underpin your findings using the lidar intensity channel.

### 4. Can you think of ways to improve your tracking results in the future?

