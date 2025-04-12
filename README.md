# SATELLITE BASED LAND COVER CLASSIFICATION SYSTEM USING AI

### Presented By
- VISHNUPRASATH K  : 6176AC21UEC160

### Under the Guidance of
- MR. STALIN C JOSE M.E (ASST. PROFESSOR)

---

## SATELLITE BASED LAND COVER CLASSIFICATION SYSTEM USING AI
<table>
  <tr>
    <td><img src="README_images/Satellite-Based-Land-Cover-Classification-Using-AI.png" width="300px"></td>
    <td><img src="README_images/satellite3.png" width="300px"></td>
  </tr>
</table>

### CONTENTS
- ABSTRACT
- INTRODUCTION
- HARDWARE REQUIREMENTS
- SOFTWARE REQUIREMENTS
- EXISTING SYSTEM
- PROPOSED SYSTEM
- METHODOLOGY
- TECHNOLOGIES USED
- FINAL OUTPUT
- PERFORMANCE ANALYSIS
- CONCLUSION

---

### ABSTRACT
- This project utilizes **satellite imagery and AI techniques** to classify land cover types efficiently.
- Combines **Reinforcement Learning (RL)** and **Key Point Matching** within **Google Earth Engine (GEE)** to analyze **Landsat satellite images**.
- Implements a novel hybrid model for **environmental monitoring**, **urban planning**, and **natural disaster analysis**.

---

### INTRODUCTION
- Satellite imagery offers vast and valuable insights into Earth's geography.
- Traditional image classification struggles with **temporal variation**, **cloud noise**, and **dimensionality**.
- The proposed method integrates **2-layer Recurrent Learning (TLRL)** and **feature-matching** for improved classification accuracy.

---

### HARDWARE REQUIREMENTS
- **Processor:** Intel Core i5 or above  
- **RAM:** 8 GB minimum  
- **GPU:** NVIDIA CUDA (optional for MATLAB GPU support)  
- **Storage:** 10 GB free space  

---

### SOFTWARE REQUIREMENTS
- MATLAB R2022a or later  
- Image Processing Toolbox  
- Deep Learning Toolbox  
- Google Earth Engine (via browser or API)  
- Landsat Image Dataset  

---

### EXISTING SYSTEM
- Traditional classifiers like SVM, Random Forest, and MLPs.
- **FAULTS**:
  1. Low adaptability to dynamic environments  
  2. Poor temporal feature extraction  
  3. Inaccurate with noisy satellite data  
  4. No integration with cloud-based tools  

---

### PROPOSED SYSTEM
- **2-Layer Recurrent Learning (TLRL)** model replaces static MLPs with spline-based activation.
- Uses **ConvNeXt** and **Vision Transformers (ViT)** for classification.
- Hybrid technique leverages **GEE**, **feature extraction**, and **reinforcement learning**.

**ADVANTAGES:**
1. High precision and recall in classification  
2. Suitable for multi-temporal datasets  
3. Integration with cloud infrastructure (GEE)  
4. Real-time application potential for disaster and environmental monitoring  

---

### METHODOLOGY

**STEP-WISE FLOW:**
1. **Preprocessing** Satellite Images  
2. **Feature Extraction** using RGB and B-spline methods  
3. **Clustering** and **Morphological Reconstruction**  
4. **Training ViT Model** on segmented data  
5. **Land Use Classification** based on trained models  

> Refer the `Main2Run.m` for complete workflow

---

### TECHNOLOGIES USED
- **MATLAB**: Core development and modeling  
- **Google Earth Engine (GEE)**: For satellite image retrieval and processing  
- **Reinforcement Learning**: To optimize classification over iterations  
- **Vision Transformer (ViT)**: Advanced deep learning model for image recognition  
- **ConvNeXt Architecture**: Pre-trained CNN model enhanced using TLRL  

---

### FINAL OUTPUT

> **Input Satellite Image**  
<img src="README_images/input.png" width="400px">

> **RGB Variants and Morphological Reconstruction**  
<img src="README_images/reconstruction.png" width="400px">

> **Final Segmented Image**  
<img src="README_images/final_segmented.png" width="400px">

> **ViT Classification Output**  
<img src="README_images/vit_output.png" width="400px">

> **Performance Metrics**  
<img src="README_images/performance_graph.png" width="400px">

---

### PERFORMANCE ANALYSIS

| Metric       | Score |
|--------------|-------|
| Accuracy     | 0.92  |
| Precision    | 0.90  |
| Recall       | 0.91  |
| F1 Score     | 0.90  |
| Specificity  | 0.93  |

---

### CONCLUSION
- This project presents a high-performance hybrid model for land cover classification using satellite images.
- Combines **deep learning**, **attention mechanisms**, and **recurrent training** to achieve **greater classification performance**.
- Demonstrates strong potential for real-world applications in **urban development**, **disaster detection**, and **resource management**.

---

### HOW TO EXECUTE

> [!NOTE]  
> MATLAB (R2022a or later) is required.  
> 
> 1. Launch MATLAB  
> 2. Open `Main2Run.m`  
> 3. Click **Run**  
> 4. Select a satellite image when prompted  
> 5. The model will process and classify the image  
> 
> Required `.mat` training data should be present in the same folder (`ViT_TrainData.mat`)

---
