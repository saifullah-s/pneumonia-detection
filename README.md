[Home](https://github.com/saifullah-s/portfolio/blob/main/README.md) &nbsp;·&nbsp; [Projects](https://github.com/saifullah-s/portfolio/blob/main/projects.md) &nbsp;·&nbsp; [Skills](https://github.com/saifullah-s/portfolio/blob/main/skills.md) &nbsp;·&nbsp; [About Me](https://github.com/saifullah-s/portfolio/blob/main/about-me.md) &nbsp;·&nbsp; [Contact](https://github.com/saifullah-s/portfolio/blob/main/contact.md) &nbsp;·&nbsp; [Resume](https://github.com/saifullah-s/portfolio/blob/main/resume.md)  

# PneumoCatcher: Automated Pneumonia Detection from Chest X-Rays Using Deep Learning  

- [Full Report](https://drive.google.com/file/d/1zTl9A2gmcG1j1wzg55Ki8rwc2RpRJpas/view?usp=drive_link)  
- [Interactive Colab Notebook](https://colab.research.google.com/drive/1qtopABKr5AHa7Jb3hkJziYnVEND_cRZC?usp=drive_link)  
- [Presentation](https://drive.google.com/file/d/1E57HFy-e0yJ1MZCmnNp3jJF00kXwiahk/view?usp=drive_link) and [Live Demo](https://drive.google.com/file/d/1KvUtloTZDsC4bkQXaMo68CVBrXoAKeHX/view?usp=drive_link)  

## Mentor  
**Dr. Cihan Dagli**  
Professor at Missouri S&T, Founder and Director of Systems Engineering Graduate Program  

## Summary
-	Developed a custom CNN model (TensorFlow) and utilized transfer learning by fine-tuning the ResNet-50 model to classify pneumonia from chest X-rays, achieving 95.6% accuracy and 97.0% F1-score.
-	Implemented Grad-CAM heatmaps for explainable AI, visualizing model decision-making.
-	Uncovered overfitting in the models and provided recommendations for enhancing predictive reliability.
-	Preprocessed large-scale medical imaging data, including grayscaling, resizing, and normalization.
-	Conducted robust evaluations with metrics like Precision, Recall, F1-score, and confusion matrix analysis.
-	Delivered findings through an interactive Colab notebook, detailed report, and live presentation.

## Description  
A deep learning project using convolutional neural networks (CNNs) to detect pneumonia in chest X-ray images. This study compared a custom-designed CNN model against fine-tuned ResNet-50 architectures. Grad-CAM heatmaps were employed for interpretability, providing visual insights into the models' decision-making.  

### Concepts:  
- Computer Vision  
- Convolutional Neural Networks (CNNs)  
- Supervised Learning (Classification task)  
- Transfer Learning (ResNet-50)  
- Explainable AI (Grad-CAM visualizations)  

### Frameworks and Libraries:  
- TensorFlow  
- Scikit-learn  
- NumPy  
- Matplotlib  
- Seaborn  

## Objective  
To develop and evaluate deep learning models for reliable pneumonia detection in chest X-rays, while also providing Grad-CAM heatmaps for interpretability.  

## Key Techniques and Skills  

### Data Preprocessing  
- Grayscaling  
- Image resizing to 180x180 pixels  
- Conversion from JPEG to HDF5, and then to NumPy arrays  
- Normalization of pixel values  
- Dataset split: 80% training, 20% testing  

### Custom CNN Architecture  
- Built with convolutional layers, pooling layers, dense layers, and dropout for efficient performance.  

### Transfer Learning with ResNet-50  
- Fine-tuned ResNet-50 models for binary classification.  
- Explored two configurations:  
  - Limited retraining of newly added layers.  
  - Extended retraining of deeper ResNet-50 layers.  

### Explainable AI  
- Grad-CAM heatmaps visualized critical regions in X-rays to interpret the models' predictions.  

### Model Evaluation  
- Metrics: Accuracy, Precision, Recall, F1-score.  
- Confusion matrices assessed false positives and negatives.  

## Results  
- The custom CNN model achieved the highest accuracy (95.6%) and F1-score (97.0%), outperforming the fine-tuned ResNet-50 models.  
- Grad-CAM heatmaps highlighted both relevant and irrelevant regions, indicating potential overfitting.  
- Recommendations include improved preprocessing, dataset expansion, and data augmentation to mitigate overfitting and improve interpretability.  
- A sample Grad-CAM heatmap is shown below:

![GradCAM](sample%20gradcam%20outputs/gradcam_10.png)
