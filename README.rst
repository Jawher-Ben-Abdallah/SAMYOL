=========
SAMYOL
=========

[![License: Apache2.0](https://img.shields.io/badge/License-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0/)



🤔 What is this?
----------------

SAMYOL is a Python library that combines an object detection model and a segmentation model. It provides a unified interface for performing object detection and segmentation tasks using different versions of the YOLO model (YOLOv6, YOLOv7, YOLOv8, or YOLO-NAS) and the Segment Anything Model.
#TODO: add descriptions/Links about YOLO models and SAM.
#TODO: add some examples (pics)


🚀 Features
------------
- Integrated object detection and segmentation capabilities
- Support for multiple versions of the YOLO model
- Flexible input options for image paths
- Easy-to-use interface for obtaining object detection predictions and segmentation masks

Quick Installation
------------------
This python library requires python>=3.11, as well as pytorch>=1.7.0 and torchvision>=0.8.1 These will be installed automatically if they are not already installed. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.


.. code:: python

  # Install SAMYOL:
  pip install SAMYOL
  # or
  conda install SAMYOL -c conda-forge
  # or 
  pip install git+https://github.com/Jawher-Ben-Abdallah/SAMYOL.git 
  # or clone the repository locally and install 
  git clone git@github.com:Jawher-Ben-Abdallah/SAMYOL.git
  cd segment-anything; pip install -e .


🚀 Getting Started
-------------------
.. code:: python

  import SAMYOL

  # Create an instance of the SAMYOL class
  # Example usage with YOLOv6
  samyol = SAMYOL(input_paths='path/to/images', model_path='path/to/yolo_model', device='cuda', version='v6')
  # Run the prediction
  masks, scores = samyol.predict()
  # Access the obtained masks and scores
  # Example usage with the first mask and its corresponding score
  mask = masks[0]
  score = scores[0]
TODO: add visualizations and such


For more detailed examples and usage instructions, please refer to the  `Examples <https://link_to_examples_dir>`__ directory.


💁 Contributing
----------------
As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see [here](.github/CONTRIBUTING.md).
#TODO: add CONTRIBUTING.md file