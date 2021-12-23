#  Sensor Fusion and Tracking - Object Detection

In this lesson, I get the ideas of how to process lidar point cloud data, and detect objects in BEV image.

## Section 1 : Compute Lidar Point-Cloud from Range Image
### Visualize range image channels (ID_S1_EX1)

![](img/range_image.png)

### Visualize lidar point-cloud (ID_S1_EX2)

![](img/lidar_point_cloud.png)

## Section 2 : Create Birds-Eye View from Lidar PCL
### Compute intensity layer of the BEV map (ID_S2_EX2) 

![](img/bev_map.png)

### Compute height layer of the BEV map (ID_S2_EX3)

![](img/height_image.png)

## Section 3 : Model-based Object Detection in BEV Image

![](img/labels_detected_objects.png)

## Section 4 : Performance Evaluation for Object Detection
### DarkNet
Precision = 0.9292604501607717, Recall = 0.9444444444444444

![](img/result0.png)

![](img/performance_metric0.png)

### DarkNet
Precision = 1.0, Recall = 1.0

![](img/result1.png)

![](img/performance_metric1.png)
