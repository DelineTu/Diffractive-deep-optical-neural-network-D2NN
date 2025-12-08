# D2NN: Machine-Learned Diffractive Optical Neural Networks with Two-Photon-Polymerization Fabrication

This repository contains the machine-learning pipeline, diffractive optical simulation, and femtosecond-laser two-photon-polymerization (TPP) fabrication workflow used to design, train, and physically realize Diffractive Deep Neural Networks (D2NNs) for all-optical inference.

### 1. Training Performance (TensorFlow / PyTorch)

The D2NN is trained end-to-end using:
- Angular Spectrum Propagation
- Negative Pearson Correlation Coefficient (NPCC) loss
- Two-layer diffractive modulation
- GPU-accelerated batches

Training converges smoothly and achieves 95% accuracy. Training curves (loss & accuracy):
<td><img src="https://cdn.jsdelivr.net/gh/DelineTu/Pictures/202512061224912.png" width="800" /></td>

### 2. Fabrication via Femtosecond-Laser Two-Photon Polymerization (TPP)

The trained diffractive layers are converted into 3D voxel height maps and directly printed using:
- Femto-second laser writing
- Two-photon polymerization (TPP)
- Sub-micron resolution
- 3D topography

Below are SEM images of the fabricated D2NN layers.
<td><img src="https://github.com/user-attachments/assets/4b4f16b8-505a-4e14-b50b-f4e2e419c19d" width="600" /></td>

### 3. Optical Inference Results

Example results include:

- Class intensity distribution at the detector plane
- Input digit image
- Diffracted output pattern
<td><img src="https://github.com/user-attachments/assets/0f9fd643-4750-45e9-8e46-ef0a99b46bcb" width="600" /></td>

#### Reference

D2NNs were first introduced by Ozcan et al. in the Science 2018 paper (https://www.science.org/doi/10.1126/science.aat8084), which established the framework of using trainable diffractive layers to implement deep optical neural networks.

https://github.com/awsomecat/Diffractive-Deep-Neural-Network

#### License
This project is licensed under the GNU General Public License v3.0. 

A copy of this license is given in this repository as [license.txt](https://github.com/DelineTu/Diffractive-deep-optical-neural-network-D2NN/blob/main/LICENSE).
