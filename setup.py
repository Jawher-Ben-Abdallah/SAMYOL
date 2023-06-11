from setuptools import setup, find_packages

with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
  name="SAMYOL_",
  version="1.0.0",
  description="Combines YOLO models and SAM",
  package_dir={"": "samyol"},
  packages=find_packages(where="samyol"),
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/Jawher-Ben-Abdallah/SAMYOL",
  author="Jawher Ben Abdallah - Rim Sleimi",
  author_email="jawher.b.abdallah@gmail.com",
  license='Apache License 2.0',
  classifiers=[
      "Programming Language :: Python :: 3.10",
    ],
    keywords="YOLOv6, YOLOv7, YOLOv8, YOLO-NAS SAM, Object_Detection, Segmentation",
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
        "onnxruntime"
    ],
    
    extras_require={"dev" : "twine>=4.0.2"},
    python_requires=">=3.10",
)
