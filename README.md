# Deep-Learning-Based-Prediction-System-for-Optimal-Embryo-Transfer-Timing-

### Overview
This project presents a deep learning–based approach for **endometrial segmentation and receptivity prediction** from ultrasound images.  
The aim is to automatically identify the endometrium, measure its **thickness**, and classify it into clinically relevant categories based on predefined medical thresholds.

This solution was developed as a **proof-of-concept**, demonstrating how **computer vision and deep learning** can assist in non-invasive reproductive health assessment.

---

## 1. Background
Endometrial receptivity is a key biomarker for successful embryo implantation.  
Its evaluation is traditionally manual, time-consuming, and subjective.  
By using a **segmentation model**, we can automatically isolate the endometrium from ultrasound scans and measure its **maximum thickness**, providing a consistent basis for interpretation.

---

## 2. Data Preparation
Due to the confidentiality of clinical datasets, images were **collected from public sources** and **manually annotated** using [Roboflow](https://roboflow.com/).

- **Annotation type:** Polygon-based segmentation masks  
- **Export format:** YOLOv8 segmentation  
- **Augmentations applied:**  
  - Horizontal flips  
  - Brightness and contrast adjustments  
- **Image resolution:** 512 × 512 pixels  

*(Dataset is not publicly shared due to medical content and copyright restrictions.)*

---

## 3. Model Architecture

The model is built using **UNet++** (Nested U-Net) with an **EfficientNet-B3 encoder** pretrained on ImageNet.  
This combination provides a strong balance between **semantic accuracy** and **computational efficiency**.

**Key components:**
- **Encoder:** EfficientNet-B3 (feature extractor)
- **Decoder:** UNet++ with deep supervision
- **Activation:** None (raw logits for numerical stability)
- **Loss Function:**  
  - **Dice Loss** — focuses on shape overlap between prediction and ground truth  
  - **Soft Binary Cross-Entropy Loss** — penalizes pixel-level misclassification  
  - Combined total loss = `Dice + 0.5 * BCE`
- **Optimizer:** Adam (learning rate = 1e-4)
- **Scheduler:** Cosine Annealing LR (smooth decay)
- **Mixed Precision Training:** Enabled with PyTorch `GradScaler` for efficiency
- **Epochs:** 20  
- **Validation Accuracy:** ~70% (on limited dataset)

---

## 4. Thickness Estimation and Clinical Classification

Once segmentation is complete, the binary mask is analyzed to estimate **endometrial thickness**.

**Thickness calculation logic:**
- Compute the Euclidean **distance transform** on the mask.
- Estimate the maximum width (`2 × max(distance)`).
- Convert from **pixels → millimeters** using a fixed calibration:  
  **1 mm ≈ 5 pixels.**

**Classification thresholds:**
| Thickness (mm) | Classification     |
|----------------|--------------------|
| < 7 mm         | Non-Receptive      |
| 7–8 mm         | Pre-Receptive      |
| 8–15 mm        | Receptive          |
| > 15 mm        | Hyper-Thickened    |

---

## 5. Deployment
The trained model was deployed using **Streamlit**, creating an intuitive web interface for real-time testing.

**Dashboard features:**
- Upload ultrasound image  
- Automatic segmentation visualization  
- Predicted thickness (in mm)  
- Receptivity classification result  

Example UI screenshots (from private testing):

| Upload Screen | Segmentation Output |
|----------------|---------------------|
| ![Upload Screenshot](https://github.com/user-attachments/assets/5743d232-9243-4827-937a-afe7038782c9) | ![Output Screenshot](https://github.com/user-attachments/assets/16a9a44a-654f-4883-be45-8b048786f970) |

*(Screenshots are representative. The actual Streamlit app is locally hosted due to privacy constraints.)*

---

## 6. Results Summary
| Metric | Description | Value |
|--------|-------------|-------|
| **Dice + BCE Loss** | Combined segmentation objective | Stable convergence |
| **Mean IoU (Validation)** | Intersection over Union for segmentation | ~0.70 |
| **Mean Absolute Percentage Error (MAPE)** | Thickness estimation accuracy | 70% |
| **Classification Accuracy** | Based on clinical thresholds | ~70% |

These results indicate that while the model performs well for a proof-of-concept, further improvements depend on the availability of high-quality, labeled medical data.

---

## 7. Technologies Used
- **Python**  
- **PyTorch**  
- **segmentation-models-pytorch (UNet++)**  
- **OpenCV**  
- **NumPy / SciPy**  
- **Streamlit** (for UI deployment)  
- **Roboflow** (for data annotation)

---

## 8. Key Learnings
This project provided hands-on experience in:
- Building segmentation architectures for medical imaging  
- Using composite loss functions (Dice + BCE)  
- Leveraging pretrained encoders for feature efficiency  
- Mixed-precision training for optimization  
- Practical deployment using Streamlit  
- Designing explainable thickness-based clinical rules  

---

## 9. Next Steps
Potential improvements include:
- Using actual clinical datasets with proper scaling metadata  
- Incorporating multi-class segmentation for adjacent structures  
- Extending classification to multi-factor receptive prediction (e.g., vascularity, pattern)

---

## 10. Acknowledgements
Special thanks to publicly available ultrasound datasets and open-source tools such as PyTorch and Roboflow, which made this experimentation possible.

---

### Author
**Puneeth M**  
Deep Learning and Computer Vision Enthusiast  
📧 [thepuneethmail@gamil.com  
🌐 [https://www.linkedin.com/in/puneeth-m-5b4345238/]  

---

> _Note: This repository contains only the documentation and results of the project. The dataset and trained model are not publicly shared due to confidentiality and ethical considerations._
![upload](https://github.com/user-attachments/assets/5743d232-9243-4827-937a-afe7038782c9)
![output](https://github.com/user-attachments/assets/16a9a44a-654f-4883-be45-8b048786f970)
