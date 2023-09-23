# Master_Project
This repository contains all relevant code and experimental results for the Master's project
# Content
This repository is designed to support all experimental methods and results for the COMP0132 report titled "Efficient Gaussian Processes Regression Representation For Reachability Map and Inverse Reachability Map." 
Within the repository, 
1. The GPR file is used for representing RM (Reachability Map) and IRM (Inverse Reachability Map) using Gaussian Processes Regression. It investigates core reseach problems such as: optimal training set size, model optimisation methods, and outlier detection.
3. The TCPPBasic-main file is used to evaluate the performance of IRM represented by GPR in the Task-consistent RRT* algorithm.
# A. Download RAW data
1. ```
   cd ~/Master_Project/
   ```

2. ```
   cd TCPPBasic-main/+RM/data/
   pip install gdown
   gdown https://drive.google.com/drive/folders/1WnWSa3K3EYJ2r4WdKejPhwp6ngsqCY58?usp=sharing -O . --folder
   gdown https://drive.google.com/drive/folders/1NdLY1aQjJqht8RryoKCXhp5464UROlza?usp=sharing -O . --folder
   ```

# B. Test the GPR 
see the _GPR_script.m_ in ~/Master_Project/GPR/
## Test for RM
```
map_flag = 0;
```
## Test for IRM
```
map_flag = 1;
```

# C. Test the TCPP
## Create the task scene
1. Find the ~/Master_Project/TCPPBasic-main/+Tasks/specific_tasks/
2. Run _HelloWorldConstant.m_
3. Select file: ~/Master_Project/m3dp_scenes/test_world/test_world.yaml
## Test the Task-consistent RRT*
1. Find the ~/Master_Project/TCPPBasic-main/+RRT/
2. Run _entry_point.m_
3. Select file: ~/Master_Project/m3dp_scenes/test_world/tasks/hello_world_constant.mat
   
 
