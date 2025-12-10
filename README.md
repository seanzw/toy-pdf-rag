# toy-pdf-rag
A toy pdf RAG example for self-learning. See the [post](https://seanzw.github.io/posts/simple-document-rag/).

### Setup & Use (Windows)

This single, step-by-step section shows how to set up and run the tool on a fresh Windows machine. Follow each numbered step exactly. If you are not sure what to click or type, follow the sub-steps.

1) Open PowerShell (the Windows terminal you'll use)
    - Method A (Start menu): Click the Windows Start button, type `PowerShell`, then click `Windows PowerShell` or `PowerShell` to open it.
    - Method B (from File Explorer): Open File Explorer, go to any folder, click the address bar, type `powershell` and press Enter — PowerShell will open in that folder.
    - To open PowerShell as administrator: Click Start, type `PowerShell`, then right-click `Windows PowerShell` and choose **Run as administrator**.

2) Install Ollama and pull the retrieval model
    - Download and install Ollama from https://ollama.com/ (choose the Windows installer). Follow the installer prompts.
    - After installation, open PowerShell and run:
```powershell
ollama pull dengcao/Qwen3-Embedding-0.6B:F16
```
This downloads the local retrieval/embedding model the project uses.

3) Install Python and add it to PATH
    - Download the latest Python 3 installer for Windows from https://www.python.org/downloads/windows/.
    - Run the installer and IMPORTANT: check the box that says **Add Python to PATH** before clicking Install.
    - If you already installed Python without checking that box, add it manually:
      - Press Start, type **Edit the system environment variables**, open it, click **Environment Variables...**, select the `Path` line under your user variables, click **Edit**, then **New** and add both (example paths — replace with your actual install path):
         - `C:\Users\YourName\AppData\Local\Programs\Python\Python3X\`
         - `C:\Users\YourName\AppData\Local\Programs\Python\Python3X\Scripts\`
      - Click OK to save.
    - Verify Python is available by opening a new PowerShell window and running:
```powershell
python --version
```

4) Open PowerShell in the project folder (where `pdf_analyze.py` is)
    - Using File Explorer: navigate to the project folder (the folder that contains `pdf_analyze.py`), click the address bar, type `powershell` and press Enter.
    - Or, in File Explorer, hold Shift, right-click the folder background and choose **Open PowerShell window here** (or **Open in Terminal**, then choose PowerShell).

5) Create and activate a Python virtual environment named `pdf-rag`
In the PowerShell window that is already in the project folder, run:
```powershell
python -m venv pdf-rag
```

Activate the venv:
```powershell
.\pdf-rag\Scripts\Activate.ps1
```

If PowerShell blocks script execution, allow local scripts (you may need to run PowerShell as administrator for this command the first time):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\pdf-rag\Scripts\Activate.ps1
```

(If you prefer Command Prompt instead of PowerShell, activate with `.
pdf-rag\Scripts\activate`.)

6) Install required Python packages
    - With the `pdf-rag` venv activated, run:
```powershell
pip install -r requirements.txt
```

7) Add your API key (optional if using local-only models)
    - Create a file named `api_key.py` in the project root (same folder as `pdf_analyze.py`).
    - Put your key into the file like this (replace the placeholder text with your real key):
```python
API_KEY = "your_api_key_here"
```

Note: If you plan to run reasoning locally with Ollama models, you do not need an API key.

How to run the analyzer (every time you use the tool)
 - 1) Start the Ollama retrieval model (keep this running):
```powershell
ollama run dengcao/Qwen3-Embedding-0.6B:F16
```
  Keep that PowerShell window open — it serves embedding requests.

 - 2) Open a new PowerShell window, navigate to the project folder, and activate the `pdf-rag` venv:
```powershell
cd C:\path\to\toy-pdf-rag
.\pdf-rag\Scripts\Activate.ps1
```

 - 3) Run the analyzer on your PDF (replace the path with your PDF location):
```powershell
python pdf_analyze.py .\materials\dsv3.pdf
```
  The tool will save JSON and Markdown outputs next to the PDF (e.g., `materials/dsv3.result_001.md`).

If anything fails, check the specific command output in PowerShell and make sure:
- You opened PowerShell in the project folder.
- The `pdf-rag` virtual environment is activated before running `pip` or `python`.
- Ollama is running the retrieval model if you rely on local embeddings.

This section aims to be explicit for new Windows users; if you'd like I can also add screenshots or convert these steps into a short script you can run.

### Example

In `queries.txt`, I prepared two questions:

```
What is the main contribution of this paper?

Does this paper compare with other models?
```

Then I run

```bash
python pdf_analyzer.py materials/dsv3.pdf
```

The answer:

```json
            "answers": [
                {
                    "question": "What is the main contribution of this paper?",
                    "answer": 1,
                    "explanation": "Yes. The main contributions include developing DeepSeek-R1-Zero through pure reinforcement learning without supervised fine-tuning (SFT), creating a pipeline for DeepSeek-R1 with RL and SFT stages, and demonstrating effective distillation of reasoning capabilities into smaller models, as detailed in the contributions section. (Page 4)",
                    "total_question_idx": 1
                },
                {
                    "question": "Does this paper compare with other models?",
                    "answer": 1,
                    "explanation": "Yes. The paper compares DeepSeek models with other models such as GPT-4o, Claude-3.5-Sonnet, OpenAI-o1-mini, QwQ-32B-Preview, and DeepSeek-V3 across benchmarks like AIME 2024, MATH-500, GPQA Diamond, and LiveCodeBench, as shown in evaluation tables and discussions. (Pages 4, 14, 15)",
                    "total_question_idx": 2
                }
            ]
```

You can find the full markdown result in `materials/dsv3.result_001.md`.