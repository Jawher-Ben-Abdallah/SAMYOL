=======
SAMYOL
=======

.. image httpsimg.shields.iobadgeLicense-Apache%202.0-blue.svg
   target httpsopensource.orglicensesApache-2.0
   alt License Apache 2.0

.. image httpsimg.shields.iogithubstarsJawher-Ben-AbdallahSAMYOL.svgstyle=social
   alt GitHub stars
   target httpsgithub.comJawher-Ben-AbdallahSAMYOLstargazers

.. image httpsgithub.comJawher-Ben-AbdallahSAMYOLactionsworkflowslint-format-install.ymlbadge.svg
   alt lint-format-install
   target httpsgithub.comJawher-Ben-AbdallahSAMYOLactionsworkflowslint-format-install.yml


🤔 What is this
----------------

SAMYOL is a Python library that combines an object detection model and a segmentation model. It provides a unified interface for performing object detection and segmentation tasks using different versions of the YOLO model (YOLOv6, YOLOv7, YOLOv8, or YOLO-NAS) and the Segment Anything Model (SAM).

🧩 Features
------------
- Integrated object detection and segmentation capabilities
- Support for multiple versions of the YOLO model
- Flexible input options for image paths
- Easy-to-use interface for obtaining object detection predictions and segmentation masks

.. raw html

   div style=overflow auto;
      img src=httpsgithub.comJawher-Ben-AbdallahSAMYOLrawmainassetsexamplesExample_1.png alt=Example 1 style=width 250px; float left; margin-right 5px;
      img src=httpsgithub.comJawher-Ben-AbdallahSAMYOLrawmainassetsexamplesExample_3.png alt=Example 3 style=width 300px; float left; margin-right 5px;
      img src=httpsgithub.comJawher-Ben-AbdallahSAMYOLrawmainassetsexamplesExample_2.png alt=Example 2 style=width 250px; float left;
   div


.. raw html

   div style=overflow auto;
      img src=httpsgithub.comJawher-Ben-AbdallahSAMYOLrawmainassetsmasksmask_1.png alt=Mask 1 style=width 250px; float left; margin-right 5px;
      img src=httpsgithub.comJawher-Ben-AbdallahSAMYOLrawmainassetsmasksmask_3.png alt=Mask 3 style=width 300px; float left; margin-right 5px;
      img src=httpsgithub.comJawher-Ben-AbdallahSAMYOLrawmainassetsmasksmask_2.png alt=Mask 2 style=width 250px; float left;
   div




⏳ Quick Installation
---------------------
SAMYOL is installed using pip. 

This python library requires python=3.11, as well as pytorch=1.7.0 and torchvision=0.8.1. Installing both PyTorch and TorchVision with CUDA support is strongly recommended. We will also need the models checkpoints. These will be installed automatically if they are not already installed.

.. code-block shell

   pip install SAMYOL


   
🚀 Getting Started
-------------------

The following notebook has detailed examples and usage instructions for each YOLO model

 .. image httpscolab.research.google.comassetscolab-badge.svg
         target httpscolab.research.google.comgithubJawher-Ben-AbdallahSAMYOLblobmainSAMYOL.ipynbauthuser=1



💁 Contributing
----------------
As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.
For detailed information on how to contribute, see `here httpsgithub.comJawher-Ben-AbdallahSAMYOLblobmainCONTRIBUTING.md`_
