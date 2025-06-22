# 🧠 Brain Tumor Detection Using CNN

This project focuses on detecting brain tumors from MRI scans using Convolutional Neural Networks (CNNs). It was done as part of a team mini-project — and for me, it became a deep dive into CNN architectures, optimizers, and medical image preprocessing.

> ⚠️ While the initial framework was inspired by [Nicknochnack](https://github.com/nicknochnack)'s tutorial on facial emotion detection, the **architecture, dataset, experimentation, and evaluation** were extended significantly for this domain-specific application.

---

## 🧪 What I Worked On

- Adapted and trained **AlexNet** and **LeNet** architectures on a labeled brain MRI dataset
- Compared performance using **Adam** and **SGD** optimizers
- Handled **image preprocessing**, normalization, and dataset splitting (train/val/test)
- Plotted training curves (accuracy/loss) and evaluated with **precision, recall, and accuracy**
- Saved and tested models on new MRI data using TensorFlow

---

## 🧠 Key Results

| Model          | Optimizer | Accuracy | Precision | Recall   |
|----------------|-----------|----------|-----------|----------|
| AlexNet        | Adam      | **94.76%** | 97.19%    | 95.05%   |
| AlexNet        | SGD       | 80.24%   | 92.07%    | 79.47%   |
| LeNet          | Adam      | 68.16%   | 68.16%    | 100%     |

✅ **AlexNet + Adam** delivered the best results — suggesting it's well-suited for this binary classification task.

### 🔬 Additional Experimentation

We also experimented with a **VGG16-based transfer learning model**, but it underperformed compared to AlexNet on our dataset — possibly due to overfitting or lack of fine-tuning for binary classification. As a result, we excluded it from our final report.

---

## 🗃 Dataset

We used a public MRI dataset containing 1547 labeled brain images (`tumor` / `no tumor`).  
Data was loaded from Google Drive and preprocessed using OpenCV and TensorFlow utilities.

Due to size constraints, the dataset is not uploaded to this repo. You can find it [here](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection).

---

## 🛠 Tech Stack

- Python, TensorFlow, Keras, OpenCV
- Google Colab for training
- Matplotlib for plotting
- cv2, os, imghdr, NumPy

---

## ▶️ How to Run

1. Upload the dataset to your Google Drive.
2. Mount Google Drive in Colab.
3. Run each block of code sequentially in the notebook.

---

## 🔧 Switching Optimizers

By default, the model uses the **Adam optimizer**, which gave us the best results.  
To try a different optimizer (like SGD), change the following line in the code:

```python
# From:
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# To:
model.compile('SGD', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
```

---

## 🤝 Shoutouts

- **Original CNN starter code**: [Nicknochnack](https://github.com/nicknochnack) — the backbone that helped me begin experimenting with CNNs.
- **Team**: Huge thanks to **Anukeerthana** and **Archana** — this project was a true team effort. We worked together on everything from model development to dataset exploration and documentation. It was a great collaborative learning journey.

---

## ✅ Conclusion

This project was my first end-to-end ML implementation experience — from dataset handling to model experimentation. It deepened my understanding of CNNs and the practical challenges of applying deep learning to real-world data.
