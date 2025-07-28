# PRSNet Project

This project is based on [OneFormer3D](https://github.com/filapro/oneformer3d) and includes custom implementations of the following modules: **Mamba**, **DAB**, **Gauss**, and **MoE**.

---

## 📦 Environment Setup

Follow the environment setup instructions provided in the original [OneFormer3D repository](https://github.com/filapro/oneformer3d).

```bash
git clone https://github.com/filapro/oneformer3d.git
cd oneformer3d
# Create and activate the conda environment
conda create -n oneformer3d python=3.8
conda activate oneformer3d
# Continue following the official installation steps
```

---

## 📁 Dataset

1. Download the dataset from [data.zip](https://drive.google.com/file/d/1aGVxobs83sUcpH4YzXv8KXcelnJnAD5t/view?usp=drive_link)
2. Unzip the file:
   ```bash
   unzip data.zip
   ```
3. Place the extracted `data` folder in the project root directory:

   ```
   oneformer3d/
   ├── data/
   └── ...
   ```

---

## 🧠 Model Weights

1. Download pre-trained models from [work_dirs.zip](https://drive.google.com/file/d/1jwmIJ2lmYJD3n-CvOAjWuooKsxK4TvMO/view?usp=sharing)
2. Unzip the file:
   ```bash
   unzip work_dirs.zip
   ```
3. Place the extracted `work_dirs` folder in the project root directory:

   ```
   oneformer3d/
   ├── work_dirs/
   └── ...
   ```

---

## 📜 Training Log

The training log is saved in:

```
result.log
```

---

## 🧩 Module Implementations

Custom implementations for the following modules can be found in the `oneformer3d` directory:

- **Mamba**
- **DAB**
- **Gauss**
- **MoE**

Refer to corresponding source files for details.

---

## ✅ Evaluation & Inference Time

To run the evaluation and check inference time, execute the following script:

```bash
bash tools/test_cal_time.sh
```

---

## 📂 Project Structure Overview

```
oneformer3d/
├── data/               # Extracted dataset
├── work_dirs/          # Extracted model checkpoints
├── tools/
│   └── test_cal_time.sh
├── result.log          # Training log
├── ...                 # Other source files
```

---

## 📮 Contact

For questions or issues, feel free to open an issue or contact the project maintainer.
