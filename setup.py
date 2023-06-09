from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
  name="SAMY",
  version="1.0.0",
  description="Combines YOLOv8 and SAM",
  package_dir={"": "samy"},
  packages=find_packages(where="samy"),
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/Jawher-Ben-Abdallah/YOLO-SAM",
  author="Jawher Ben Abdallah - Rim Sleimi",
  author_email="jawher.b.abdallah@gmail.com - sleimi.rim1996@gmail.com",
  license='Apache License 2.0',
  classifiers=[
      "License :: OSI Approved :: Apache 2.0 License",
      "Programming Language :: Python :: 3.10",
    ],
    keywords="YOLOv8, SAM, Object_Detection, Segmentation",
    install_requires=[
        "matplotlib>=3.2.2",
        "numpy>=1.21.6",
        "opencv-python>=4.6.0",
        "Pillow>=7.1.2",
        "PyYAML>=5.3.1",
        "requests>=2.23.0",
        "scipy>=1.4.1",
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "tqdm>=4.64.0",
        "ultralytics",
    ],
    extras_require={"dev" : "twine>=4.0.2"},
    python_requires=">=3.10",
)
