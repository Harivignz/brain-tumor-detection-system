# 🧠 Brain Tumor Detection, Segmentation & Stage Classification using Deep Learning

This project uses deep learning to perform automatic **brain tumor classification**, **3D segmentation**, and **stage prediction** using **MRI scans**. It's built with TensorFlow, PyTorch, and a custom-trained pipeline.

---

## 🚀 Features

- ✅ Detect tumor presence (Yes/No)
- ✅ Segment tumor region using a 3D U-Net
- ✅ Predict tumor stage (Grade I–IV)
- ✅ Final app exportable as `.exe` for Windows
- ✅ Modular architecture using CNN and PyTorch models

---

## 🗂️ Folder Structure
brain-tumor-detection-system/
├── app/ # Main executable / GUI entry
├── models/ # Trained .h5 and .pth models
├── data/ # Numpy data, labels
├── outputs/ # Visual results or logs
├── src/ # Training and testing scripts
├── README.md
├── requirements.txt
└── .gitignore


---

## 💾 Installation

```bash
git clone https://github.com/your-username/brain-tumor-detection-system.git
cd brain-tumor-detection-system
pip install -r requirements.txt


| Task                 | Framework  | Model Used             |
| -------------------- | ---------- | ---------------------- |
| Tumor Classification | TensorFlow | Custom 3D CNN (HDF5)   |
| Tumor Segmentation   | PyTorch    | 3D U-Net               |
| Stage Classification | PyTorch    | Custom 3D CNN (Grades) |



🛠️ How to Run


# For model testing
python src/test_cnn.py          # Tumor classification
python src/test_unet.py         # Tumor segmentation
python src/test_stage_cnn.py    # Tumor stage prediction

# Or run the full app:
python app/main.py


🧪 Dataset
Dataset used: https://www.med.upenn.edu/cbica/brats2021/

Format: Preprocessed .npy files

📜 License
MIT License - use freely with citation.

---

## ✅ Step 4: Create `.gitignore`

In your project root:
.gitignore
pycache/
*.pyc
*.pyo
*.pyd
*.DS_Store
*.h5
*.pth
*.npy
*.log
outputs/
*.exe
build/
dist/
.env/
.venv/



---

## ✅ Step 5: Create `requirements.txt`

You can auto-generate it using:

```bash
pip freeze > requirements.txt

or

tensorflow==2.19.0
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
numpy
matplotlib
scikit-learn
h5py
PyQt5       # Only if GUI used. 
