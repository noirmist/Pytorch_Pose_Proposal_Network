# Pytorch_Pose_Proposal_Network
Real-time human pose estimation code based on pose proposal network(ECCV2018)

# Webcam Test Result
<img src="https://imgur.com/BHTcH5x.gif" width="450px" height="300px" alt="Test result"></img><br/>

# Difference from Pose Proposal Network(T. Sekii- ECCV 2018)
## Network - Dilated Residual Network
<img src="https://imgur.com/ilnQyxU.jpg" width="500px" alt="Network image "></img><br/>

## Training - GradNorm
<img src="https://imgur.com/y5CzBPF.jpg" width="500px" alt="GradNorm"></img><br/>

# Environment
```
OS: Ubuntu 16.04
Python version: 3.5.2
PyTorch version: 1.1.0
```

# How To Use

1. Data preprocessing by using code in data folder
2. Update directory paths in train.sh
3. Train the network
4. Update directory paths in rt_test.sh and run the code
