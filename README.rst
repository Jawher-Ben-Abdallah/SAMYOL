=======
SAMYOL
=======
.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License: Apache 2.0

.. image:: https://img.shields.io/github/stars/Jawher-Ben-Abdallah/SAMYOL.svg?style=social
   :alt: GitHub stars
   :target: https://github.com/Jawher-Ben-Abdallah/SAMYOL/stargazers



🤔 What is this?
----------------

SAMYOL is a Python library that combines an object detection model and a segmentation model. It provides a unified interface for performing object detection and segmentation tasks using different versions of the YOLO model (YOLOv6, YOLOv7, YOLOv8, or YOLO-NAS) and the Segment Anything Model.


.. raw:: html

   <div style="overflow: auto;">
      <img src="https://github.com/Jawher-Ben-Abdallah/SAMYOL/raw/main/assets/examples/Example_1.jpeg" alt="Example 1" style="width: 250px; float: left; margin-right: 5px;">
      <img src="https://github.com/Jawher-Ben-Abdallah/SAMYOL/raw/main/assets/examples/Example_3.jpeg" alt="Example 3" style="width: 300px; float: left; margin-right: 5px;">
      <img src="https://github.com/Jawher-Ben-Abdallah/SAMYOL/raw/main/assets/examples/Example_2.png" alt="Example 2" style="width: 250px; float: left;">
   </div>


.. raw:: html

   <div style="overflow: auto;">
      <img src="https://github.com/Jawher-Ben-Abdallah/SAMYOL/raw/main/assets/masks/mask_1.jpeg" alt="Mask 1" style="width: 250px; float: left; margin-right: 5px;">
      <img src="https://github.com/Jawher-Ben-Abdallah/SAMYOL/raw/main/assets/masks/mask_3.jpeg" alt="Mask 3" style="width: 300px; float: left; margin-right: 5px;">
      <img src="https://github.com/Jawher-Ben-Abdallah/SAMYOL/raw/main/assets/masks/mask_2.png" alt="Mask 2" style="width: 250px; float: left;">
   </div>


🧩 Features
------------
- Integrated object detection and segmentation capabilities
- Support for multiple versions of the YOLO model
- Flexible input options for image paths
- Easy-to-use interface for obtaining object detection predictions and segmentation masks


⏳ Quick Installation
---------------------
SAMYOL is installed using pip. 

This python library requires python>=3.11, as well as pytorch>=1.7.0 and torchvision>=0.8.1. Installing both PyTorch and TorchVision with CUDA support is strongly recommended. We will also need the models checkpoints. These will be installed automatically if they are not already installed.

.. code:: python

  # Install SAMYOL:
  pip install SAMYOL
  # or
  conda install SAMYOL -c conda-forge
  # or 
  pip install git+https://github.com/Jawher-Ben-Abdallah/SAMYOL.git 
  # or clone the repository locally and install 
  git clone git@github.com:Jawher-Ben-Abdallah/SAMYOL.git
  cd segment-anything; 
  pip install -e .


🚀 Getting Started
-------------------

The following notebook has detailed examples and usage instructions for each YOLO model:

 .. image:: https://colab.research.google.com/assets/colab-badge.svg
         :target: https://github.com/Jawher-Ben-Abdallah/SAMYOL/blob/main/SAMYOL.ipynb

#TODO: Fix the link

💁 Contributing
----------------
As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.
#TODO: Fix the link
For detailed information on how to contribute, see [here](https://github.com/Jawher-Ben-Abdallah/SAMYOL/blob/main/CONTRIBUTING.md).
