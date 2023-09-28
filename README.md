**README**

This guide will help you convert and benchmark ONNX models using OpenVINO. Follow the steps below:

1. Install OpenVINO by referring to the official installation guide: [Installing OpenVINO with pip](https://docs.openvino.ai/2023.1/openvino_docs_install_guides_installing_openvino_pip.html).

2. Convert the ONNX model to the OpenVINO format. Run the following command:
   ```
   python convert_onnx2openvino.py --onnx-path <your onnx path> --output-path <your output openvino file path>
   ```
   Replace `<your onnx path>` with the path to your ONNX model, and `<your output openvino file path>` with the desired output path for the converted model.

3. Measure the model's performance using OpenVINO by running the benchmark_app. Execute the following command:
   ```
   benchmark_app -m <your openvino>.xml -d CPU -hint latency
   ```
   Replace `<your openvino>` with the path to your OpenVINO model XML file.

Please refer to the official OpenVINO documentation for more detailed instructions and troubleshooting.


**Testing Platform:**
```
Distributor ID: Ubuntu
Description: Ubuntu 20.04.6 LTS
Release: 20.04
Codename: focal
```

**CPU:**
```
12th Gen Intel(R) Core(TM) i7-12700K
```

ResNet50 Models:

| Model             | Accuracy | Duration (ms) | Median Latency (ms) | Average Latency (ms) | Min Latency (ms) | Max Latency (ms) | Throughput (FPS) | FLOPs (G)   | Params (M)    |
|-------------------|----------|---------------|---------------------|----------------------|------------------|------------------|------------------|-------------|---------------|
| resnet50-origin   | 76.15%   | 60016.11      | 9.57                | 10.29                | 9.22             | 28.45            | 96.26            | 4.111414272 | 25.557032     |
| resnet101_origin  | 77.37%   | 60035.02      | 17.92               | 18.65                | 17.37            | 30.93            | 53.35            | 7.83387136  | 44.54916      |
| resnet50_78top1   | 78.09%   | 60017.66      | 8.51                | 8.91                 | 8.18             | 18.25            | 111.17           | 3.3849585   | 21.810275     |
| resnet50_75top1   | 75.17%   | 60003.73      | 3.53                | 3.71                 | 3.26             | 10.87            | 263.25           | 1.205455944 | 9.097625      |
| resnet50_76top1   | 76.24%   | 60009.03      | 4.07                | 4.28                 | 3.82             | 12.17            | 228.85           | 1.41260609  | 11.144143     |

You can download all the ONNX models from the following link: [ONNX Models](https://drive.google.com/drive/folders/1EeuCOGs7gt_3vTu-LTp3beg_T0kxVOr7?usp=drive_link)