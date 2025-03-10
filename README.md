# Hair Type Classification

## 📌 About This Project
This application uses an advanced deep learning model to classify hair types based on uploaded images. Developed by **Team Visionaries**, our goal is to merge AI and beauty, providing AI-powered hair analysis for everyone. Built with **Streamlit** and **TensorFlow**.

## Project Structure
```
📂 Experimental_Models        # Collection of experimental models (CNN, VGG, DenseNet, etc.) used for testing different architectures and improving accuracy.
📂 NewTest_dataset            # Folder containing test images to evaluate model performance.
📄 advanced_densenet_model.h5 # Pre-trained DenseNet model
🐍 HairTypes_InterfaceWeb.py  # Web interface script (Streamlit-based)
📓 TheBestModel_DenseNet121.ipynb # Jupyter notebook for training
```

## 📥 Download the Model
The pre-trained DenseNet model (advanced_densenet_model.h5) can be downloaded from Google Drive:

🔗 [Download Model from Google Drive](https://drive.google.com/file/d/1YMIb1eJPndqXFQGNbr9ESimH_Uo9uPjh/view?usp=sharing)


After downloading, place it in the root directory of the project.

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
   - Place test images in `NewTest_dataset`.
   - Modify the evaluation script to load and analyze the new test data.
   - Run the model on the test dataset and review performance metrics.
