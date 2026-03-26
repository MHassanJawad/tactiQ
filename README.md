# RT-FSAS: Real-Time Football Strategy Analysis System
A Multimodal AI-powered platform for live football coaching support.

## 🤝 For Team Members: Initial Setup Guide

Welcome to the RT-FSAS repo! Since we don't track massive dataset files or bulky PyTorch libraries in Git, you need to set up your local workspace by following these **3 quick steps**.

### Step 1: Install Python Dependencies
You need a virtual environment so we don't break your system Python. Run this in your terminal:
```powershell
python -m venv rt-fsas-env
.\rt-fsas-env\Scripts\activate   # (On Mac/Linux: source rt-fsas-env/bin/activate)
```

Then, install all the exact libraries we are using for the ML models, FAISS indexing, and Gemini AI:
```powershell
pip install --upgrade pip
# Install the exact PyTorch CPU version to prevent massive CUDA bloatware
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
# Install everything else
pip install -r setup/requirements.txt
```

### Step 2: Download the Core Football Data
We are using the **StatsBomb Open Data** for this project (specifically La Liga 2015/16). Because it contains thousands of JSON match files, we do NOT push it to GitHub. 

You need to download the dataset yourself:
1. Go to **[StatsBomb Open Data GitHub](https://github.com/statsbomb/open-data)**.
2. Click the green `<> Code` button and select **Download ZIP**.
3. Extract that `.zip` file.
4. Rename the extracted folder to exactly `open-data` and place it in the direct root folder of this project.

*Note: The `.gitignore` file guarantees this huge folder will never be accidentally committed back to GitHub.*

### Step 3: Run the Verification Script
To prove everything is working perfectly, run:
```powershell
python setup/verify_imports.py
```
If you see all green checks (`OK`), you are ready to code! Refer to `PROJECT_TASKS.md` to see what tasks are assigned to you today.
