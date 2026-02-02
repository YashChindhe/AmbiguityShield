# AmbiguityShield: VLA Action-Performer Auditor

### **Project Goal**
This project provides an automated quality gate for robotics data. When a performer (human hand or robot) records a task, we need to know: **"Is this action clear enough for a model to learn from?"** **AmbiguityShield** acts as a supervisor. It evaluates the performer's action against the instruction. If the action is confusing with respect to instruction, the system **rejects** the annotation so it can be recorded again. Suppose we have 10 manual annotators working for us then the annotation manager need not evaluate each and every video with its annotation. This tool can help them to simplify their work. If 1000 videos are annotated every day and passed through this tool, and out of them if 150 fail then the manager needs to be concerned only about those 150 instead of 1000, which drastically reduces human fatigue and error.

---

## 1. Methodology: How we Audit Actions

The system evaluates the performer using a frame-by-frame analysis:

1.  **Logit Extraction**: As the video is processed, the VLA model generates **Logits** for every frame. These represent the raw probability scores for all possible next actions.
2.  **Entropy Analysis**: 
    * We calculate **Shannon Entropy** from these Logits. 
    * This measures the "uncertainty" of the model.
    * **Low Entropy**: The performer's action is clear and easy to follow.
    * **High Entropy**: The performer's action is confusing, or the objects in the scene are ambiguous.
3.  **Data Logging**: The entropy results for every frame are stored and then visualized.
4.  **Final Analysis**: The system calculates the **Average Entropy** and identifies **Spikes** to decide if the annotation is accepted or rejected.

---

## 2. The "Action Zone" Logic

The audit focuses only on the **middle 60%** of the video (the Action Zone). This ensures the "Pass/Reject" verdict is based on the actual task execution, ignoring the setup and reset time at the beginning and end of the recording.

---

## 3. Results & Thresholds

We used 4 specific sample videos recorded by myself in the `/data` folder to establish these thresholds:

| Sample Video | Scene Problem | Avg Entropy | Peak Spike | Audit Result |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Clear** | None (Clear action) | **< 2.0** | **< 2.0** | **✅ PASS** |
| **Spatial Ambiguity** | Two identical targets | **> 2.0** | **> 3.0** | **🚩 REJECT** |
| **Semantic Ambiguity** | Wrong object targeted | **> 2.0** | **> 3.0** | **🚩 REJECT** |
| **Cluttered Scene** | High visual noise | **> 2.0** | **> 3.0** | **🚩 REJECT** |

Due to GPU limits, the project could not be evaluated extensively. Currently am scaling down the OpenVLA model and then using it. If you have access to better GPU's then please evaluate it on the 16bit model instead of 4bit one.

---

## 4. How to Run on Google Colab

1.  **Upload**: Upload `AmbiguityShield.ipynb` to [Google Colab](https://colab.research.google.com/).
2.  **Hardware**: Change runtime type to **T4 GPU**.
3.  **Run**: Execute all cells. This clones the repository, installs dependencies, and loads the model.
4.  **Access UI**: Click the link generated in the final cell output.
5.  **Evaluate**: Upload a video from the `/data` folder or any other video you want to test from your pc and see the audit results.

> **Note on Performance:** When you click "Initialize Model" in the UI, please allow **3-5 minutes**. The system is downloading the 7-Billion parameter OpenVLA model and quantizing it from 16-bit to 4-bit to fit into the GPU memory. Once initialized, the model stays in memory for the whole session. Individual video processing usually takes only **30 seconds**.

---

## 5. Future Work: Automated Pipeline
The next phase is to turn this into a **Continuous Evaluation Pipeline**:
* **Automatic Evaluation**: Annotators would collect their final work somewhere in a repo or folder locally and then using connectors we can pull it into the pipeline for auditing.
* **Smart Filtering**: The system will automatically separate "Clean" data from "Rejected" data, creating a high-quality dataset without manual oversight.

---
**Summary:** AmbiguityShield uses VLA model uncertainty (Entropy) to automatically reject unclear robot training data.
