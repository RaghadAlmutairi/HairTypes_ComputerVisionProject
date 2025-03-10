# Hair Type Classification

## 📌 About This Project
This application uses an advanced deep learning model to classify hair types based on uploaded images. Developed by **Team Visionaries**, our goal is to merge AI and beauty, providing AI-powered hair analysis for everyone. Built with **Streamlit** and **TensorFlow**.

## Project Structure
```
📂 Experimental_Models        # Experimental models (CNN, VGG, DenseNet, etc.)
📂 NewTest_dataset            # Test dataset
📄 advanced_densenet_model.h5 # Pre-trained DenseNet model
🐍 HairTypes_InterfaceWeb.py  # Web interface script (Streamlit-based)
📓 TheBestModel_DenseNet121.ipynb # Jupyter notebook for training
```

## 🚀 Usage
1. **Train the Model:**
   ```bash
   jupyter notebook TheBestModel_DenseNet121.ipynb
   ```
2. **Run Web Interface (Streamlit):**
   ```bash
   streamlit run HairTypes_InterfaceWeb.py
   ```
3. **Test with New Dataset:**
   Place images in `NewTest_dataset`.


