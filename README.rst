=======
SAMYOL
=======
#TODO: Fix the link

[![License: Apache2.0](https://img.shields.io/badge/License-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0/)

[![GitHub star chart](https://img.shields.io/github/stars/Jawher-Ben-Abdallah/SAMYOL?style=social)](https://star-history.com/#Jawher-Ben-Abdallah/SAMYOL)


🤔 What is this?
----------------

SAMYOL is a Python library that combines an object detection model and a segmentation model. It provides a unified interface for performing object detection and segmentation tasks using different versions of the YOLO model (YOLOv6, YOLOv7, YOLOv8, or YOLO-NAS) and the Segment Anything Model.

#TODO: Fix the link

.. grid:: 3x1

   .. figure:: https://bitbucket.org/rim-and-jawher/samyol/src/main/assets/examples/Example_1.jpeg

   .. figure:: https://bitbucket.org/rim-and-jawher/samyol/src/main/assets/examples/Example_2.jpeg

   .. figure:: https://bitbucket.org/rim-and-jawher/samyol/src/main/assets/examples/Example_3.jpeg



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
         :target: https://colab.research.google.com/github/Rim-chan/SAMYOL.ipynb

#TODO: Fix the link

💁 Contributing
----------------
As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.
#TODO: Fix the link
For detailed information on how to contribute, see [here](https://bitbucket.org/rim-and-jawher/samyol/src/main/CONTRIBUTING.md).
