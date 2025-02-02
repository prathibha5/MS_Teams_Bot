# MS_Teams_Bot

# Project Setup and Execution Guide

## Python Version Required
Ensure you have Python **3.11.9** installed on your system.

## Setting Up the Virtual Environment

### 1. Create the Virtual Environment
```sh
python3 -m venv myenv
```

### 2. Activate the Virtual Environment
- On **macOS/Linux**:
  ```sh
  source myenv/bin/activate
  ```
- On **Windows (Command Prompt)**:
  ```sh
  myenv\Scripts\activate
  ```
- On **Windows (PowerShell)**:
  ```sh
  myenv\Scripts\Activate.ps1
  ```

## Navigate to the `src` Directory
```sh
cd src
```

## Install Dependencies
```sh
pip install -r requirements.txt
```

## Train the Model
Execute the `train_model.py` file to train and save the model:
```sh
python train_model.py
```

## Run the Application
Execute `app.py`:
```sh
python app.py
```

## Deactivate the Virtual Environment
```sh
deactivate
```

