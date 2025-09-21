# 📝 Confidence-Based Document QA Extraction with Feedback Logging and Web Demo

This project is a prototype **question answering (QA) system** built as part of an internship technical task.  
It uses a pre-trained Hugging Face extractive QA model to answer questions from document-like text.  
The system includes **confidence-based filtering**, **feedback logging**, and an interactive **Gradio web app**.

---

## 🚀 Features
- Extracts answers from text using a pre-trained BERT QA model  
- Returns **answer + confidence score**  
- Flags low-confidence predictions (default threshold = `0.75`)  
- Saves flagged outputs to `feedback_log.jsonl`  
- Allows manual corrections → stored in `corrected_data.jsonl`  
- Web interface (Gradio) for interactive testing and corrections  

---

## 📂 Project Structure
```
confidence_qa/
├── app.py # Web app (Gradio)
├── qa_pipeline.py # Backend QA pipeline with threshold + logging
├── test_model.py # Quick test script
├── requirements.txt # Dependencies
├── feedback_log.jsonl # Auto-generated flagged outputs
├── corrected_data.jsonl # Manually corrected answers
└── README.md # Project documentation
```

---

## ⚙️ Installation & Setup

1. Clone the repo:
```
   git clone https://github.com/alex-dev/confidence_qa.git
   cd confidence_qa
```

2. Create and activate virtual environment:
```
python -m venv .venv
source .venv/bin/activate   # macOS/Linux 
 ``` 
3. Install dependencies:
```
 pip install -r requirements.txt
```

