<div align="center">
  <img src="https://github.com/Hamed-Aghapanah/yolo8-based-aircraft-detection/blob/main/yolo8_airplane.gif" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">Hamed Aghapanah </font></b>
    <sup>
      <a href="https://HamedAghapanah.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">HamedAghapanah platform</font></b>
    <sup>
      <a href="https://HamedAghapanah.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>


 
[![Documentation](https://readthedocs.org/projects/mmaction2/badge/?version=latest)](https://mmaction2.readthedocs.io/en/latest/)
[![actions](https://github.com/open-mmlab/mmaction2/workflows/build/badge.svg)](https://github.com/open-mmlab/mmaction2/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmaction2/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmaction2)
[![PyPI](https://img.shields.io/pypi/v/mmaction2)](https://pypi.org/project/mmaction2/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)

%[📘Documentation](https://mmaction2.readthedocs.io/en/latest/) |
#[🛠️Installation](https://mmaction2.readthedocs.io/en/latest/install.html) |
[👀Model Zoo](https://mmaction2.readthedocs.io/en/latest/modelzoo.html) |
[🆕Update News](https://mmaction2.readthedocs.io/en/latest/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmaction2/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmaction2/issues/new/choose)

</div>

English | [简体中文](/README_zh-CN.md)
<div align="left">


## Introduction
The identification of airplanes in satellite images holds significant importance, particularly in airport environments where aircraft are in constant transit. Manual airplane detection is characterized by being time-consuming, expensive, and prone to errors. Therefore, the utilization of intelligent airplane detection through artificial intelligence presents a promising solution. This study investigates the performance of the YOLO model for airplane detection in satellite images. The model achieves a detection accuracy of 93.3%, evaluated over 350 images collected through Google Earth remote sensing. The incorporation of the attention mechanism into the YOLO model significantly enhances airplane detection accuracy. Specifically, the spatial attention mechanism focuses on the airplane's position in the image, resulting in an increased accuracy of 96.6%. The channel attention mechanism, emphasizing features in different image channels, elevates accuracy to 96.3%. The introduction of the recurrent attention mechanism, which considers temporal features, leads to an accuracy boost of 95.7%. The integration of all three attention mechanisms culminates in an impressive accuracy of 97.8%. However, it is crucial to note that the integration of the attention mechanism into the YOLO model comes at the cost of increased processing time. Despite the substantial improvement in detection accuracy, this study reveals a 20% decrease in Frames Per Second (FPS) due to the additional computational load associated with the attention mechanism. This trade-off may pose limitations for applications requiring rapid image processing. In summary, the inclusion of attention mechanisms improves the precision of airplane detection in satellite images. However, it is crucial to carefully evaluate the impact on processing time. Additionally, the code for implementing these mechanisms can be found on GitHub.
## Installation

MMAction2 depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection) (optional), and [MMPose](https://github.com/open-mmlab/mmdetection)(optional).
Below are quick steps for installation.
Please refer to [install.md](docs/install.md) for more detailed instruction.

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip3 install -e .
```

## Get Started

## Supported Datasets

<details open>
<summary>Install</summary>

Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

[![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

```bash
pip install ultralytics
```

For alternative installation methods including [Conda](https://anaconda.org/conda-forge/ultralytics), [Docker](https://hub.docker.com/r/ultralytics/ultralytics), and Git, please refer to the [Quickstart Guide](https://docs.ultralytics.com/quickstart).

</details>

<details open>
<summary>Usage</summary>

### CLI

YOLOv8 may be used directly in the Command Line Interface (CLI) with a `yolo` command:

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

`yolo` can be used for a variety of tasks and modes and accepts additional arguments, i.e. `imgsz=640`. See the YOLOv8 [CLI Docs](https://docs.ultralytics.com/usage/cli) for examples.

### Python

YOLOv8 may also be used directly in a Python environment, and accepts the same [arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI example above:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
```

See YOLOv8 [Python Docs](https://docs.ultralytics.com/usage/python) for more examples.

</details> 
## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{  paper Analyzing the Impact of Attention Mechanisms on Airplane Detection Accuracy in Satellite Images using YOLO Model
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
 
