# SSCADP

**Semi-supervised Classification of Data Streams Based on Adaptive Density Peak Clustering**

---

## 📌 Project Overview

This repository contains the source code for the paper:

**Semi-supervised Classification of Data Streams Based on Adaptive Density Peak Clustering (SSCADP)**

The project implements:
- Data stream classification with adaptive density peak clustering
- Concept drift detection
- Semi-supervised learning with incremental model updates
- Fast clustering-based change point detection

The goal is to address **label scarcity** and **non-stationary environments** in real-time data streams.

---

## 📁 Repository Structure

```
│── Electricity.csv                 # Example data stream
│── solution.py                     # Main program entry point
│── concept_drift_detect.py         # Concept drift detection logic
│── change_point.py                 # Change-point detection module
│── rho_multi_delta.py              # Adaptive density & clustering
│── README.md                       # Project documentation
```

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

### 2️⃣ Run the Main Program

```bash
python solution.py
```

By default, the program loads:

```python
pd.read_csv('Electricity.csv', header=None, sep=',')
```

You can replace it with any customized streaming dataset.

---



## 📄 Citation

If you use this repository, please cite the original paper:

> Liu, C., Wen, Y., & Xue, Y. (2020). *Semi-supervised Classification of Data Streams Based on Adaptive Density Peak Clustering*. In **ICONIP 2020: Neural Information Processing** (pp. 639–650). Springer. https://doi.org/10.1007/978-3-030-63833-7_54

### 📚 BibTeX

```bibtex
@inproceedings{liu2020sscadp,
  title={Semi-supervised Classification of Data Streams Based on Adaptive Density Peak Clustering},
  author={Liu, Changjie and Wen, Yimin and Xue, Yun},
  booktitle={ICONIP 2020: Neural Information Processing},
  pages={639--650},
  year={2020},
  organization={Springer},
  doi={10.1007/978-3-030-63833-7_54}
}
```

---

If you find this repository useful, please ⭐ **Star** it to support the project!
