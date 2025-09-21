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

