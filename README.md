# toy-pdf-rag
A toy pdf RAG example for self-learning. See the [post](https://seanzw.github.io/posts/simple-document-rag/).

### Prerequsits

* [ollama](https://ollama.com/) to run model on local machine (with GPU).
* `ollama pull dengcao/Qwen3-Embedding-0.6B:F16` to get the retrieval model.
* LangChain.
* (Optional) LLM API Key to use online model to analyze the documents. The code use `ChatDeepSeek` but you can switch to
* ChatGPT or other model.

### How to use

* Install all the prerequists.
* Create an `api_key.py` and add your key to it. It will be used in `pdf_analyze.py` to create the agent.
  **Note**: If you are using a local reasoning model (e.g. `deepseek-r1:8b` on `ollama`), you don't need an API key.
* Edit `queries.txt` with your questions. Questions is separated by empty lines. Questions can be grouped into sections.
Questions within the same section are processed together, sharing the same context so that they can relate to
each other. Sections are separated by '###'.
* Run `python pdf_analyze.py {pdf_to_process}`
* It will generate the answer in the same path as the pdf, both in json and markdown format. Also, it provides AI's reasoning process and explanation.

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