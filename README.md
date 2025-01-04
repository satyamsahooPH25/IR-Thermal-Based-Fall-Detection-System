# IR Thermal-Based Fall Detection System

## Demo Video
[Watch Demo Video](https://drive.google.com/file/d/17YBwep4HXVKkWZCNCKmd2EakihRSmod-/view?usp=drive_link)

https://drive.google.com/uc?export=preview&id=17YBwep4HXVKkWZCNCKmd2EakihRSmod-

# IR Thermal-Based Fall Detection System

## Demo Video
Watch the demo of the fall detection system below:

<iframe src="https://drive.google.com/uc?export=preview&id=17YBwep4HXVKkWZCNCKmd2EakihRSmod-" width="640" height="360" allow="autoplay"></iframe>

## Overview
Developed an infrared (IR) thermal-based fall detection system tailored for elderly individuals. The system focuses on accurate detection of fall events by extracting critical features like joint angles and inter-joint distances from thermal video feeds. Multiple machine learning models were ensembled to ensure robust classification performance.

## Key Features
- **Feature Extraction**: Extracted joint angles and inter-joint distances from IR thermal data.
- **Static Heat Signature Removal**: Designed an algorithm to segment common intensity pixel ranges at fixed coordinates, effectively eliminating static heat signatures.
- **Preprocessing Techniques**: Enhanced joint detection accuracy through:
  - **HSV Preprocessing**
  - **ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)**
  - **FastNlMeansDenoising**

## Machine Learning Models
Trained and ensembled the following machine learning models for fall/no-fall classification:
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost
- Support Vector Machine (SVM)

The ensemble was built using **AdaBoost**, achieving an **F1 score of 0.83**.

## System Workflow
1. **Thermal Video Preprocessing**:
    - Applied HSV-based preprocessing.
    - Used ESRGAN and FastNlMeansDenoising for noise reduction and feature enhancement.
2. **Static Heat Signature Removal**:
    - Segmented thermal frames to isolate and remove common intensity pixel ranges at fixed coordinates.
3. **Feature Extraction**:
    - Detected joints and calculated joint angles and inter-joint distances.
4. **Model Training**:
    - Trained individual models on the extracted features.
    - Combined predictions through AdaBoost for robust classification.

## Results
- Achieved an **F1 score of 0.83**, ensuring reliable differentiation between fall and no-fall events.





