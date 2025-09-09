# Hand-Detection Pong

This project is a **real-time Pong game controlled by hand detection** using a custom **Single Shot Detector (SSD)** object detection model. The model was trained on a custom dataset of hand images and integrated into a Pong clone so the paddle follows your hand movement via webcam input.

---

## Features

* **Custom SSD model** built from scratch using NumPy, then optimized in PyTorch.
* **Real-time hand detection** via webcam with bounding box overlay.
* **Interactive Pong game** where the paddle is controlled by your hand’s vertical position.
* **Custom dataset** of hand images labeled in Pascal VOC XML format.
* **Training pipeline** including anchor generation, loss computation, and backpropagation.

---

## Project Structure

```
├── ObjectDetection/
│   ├── SSD.py/ 
│   ├── Prepare_Data.py/ 
│   ├── Activations.py/         
│   ├── Loss_and_Optimizer.py/      
│   ├── Model_Testing.py/           
│   ├── Layers.py/
│   ├── anchor_boxes.npy/
│   └── hand_data.pkl/
├── Pong.py/            # Game mechanics of pong 
├── Pytorch.py/         # Pytorch version of SSD model
├── requirements.txt    # Python dependencies
├── README.md           # Project overview
└── .gitignore          # Ignored files (venv, checkpoints, datasets, etc.)
```

---

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/BronsonFarque/pong-hand-tracker.git
   cd pong-hand-tracker
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Run the Pong game with live detection:

   ```bash
   python Pong.py
   ```

Your webcam will open and the paddle will move up/down based on your detected hand.

