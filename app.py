# app.py
import json
import time
from pathlib import Path
import gradio as gr
from qa_pipeline import QAPredictor, CONF_THRESHOLD_DEFAULT

LOG_PATH = Path("feedback_log.jsonl")
CORR_PATH = Path("corrected_data.jsonl")

_predictor = None
def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = QAPredictor()
    return _predictor

def predict(context: str, question: str, threshold: float):
    qa = get_predictor()
    res = qa.answer_one(question, context, threshold)
    return res.predicted_answer, round(res.confidence, 4), res.flagged

def save_correction(context_id, question, predicted_answer, confidence, flagged, corrected_answer):
    corrected_answer = (corrected_answer or "").strip()
    if not corrected_answer:
        return "No correction provided."
    entry = {
        "context_id": context_id or f"manual_{int(time.time())}",
        "question": question,
        "predicted_answer": predicted_answer,
        "confidence": confidence,
        "flagged": flagged,
        "corrected_answer": corrected_answer
    }
    with CORR_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return "✅ Correction saved."

with gr.Blocks(title="Confidence-based Document QA") as demo:
    gr.Markdown("# Confidence-Based Document QA (Flagging + Feedback)")
    with gr.Row():
        context = gr.Textbox(label="Context (paste a paragraph/document)", lines=12, placeholder="Paste your document text here…")
        question = gr.Textbox(label="Question", lines=2, placeholder="Ask something about the context…")
    threshold = gr.Slider(0.1, 0.99, value=CONF_THRESHOLD_DEFAULT, step=0.01, label="Confidence threshold")
    answer_btn = gr.Button("Get Answer")

    answer = gr.Textbox(label="Predicted answer", interactive=False)
    confidence = gr.Number(label="Confidence (0–1)", interactive=False)
    flagged = gr.Checkbox(label="Flagged (below threshold)", interactive=False)

    answer_btn.click(predict, [context, question, threshold], [answer, confidence, flagged])

    gr.Markdown("### Optional: Submit a correction to build labeled data")
    context_id = gr.Textbox(label="Context ID (optional)")
    corrected = gr.Textbox(label="Corrected answer")
    save_btn = gr.Button("Save correction")
    status = gr.Markdown()

    save_btn.click(save_correction, [context_id, question, answer, confidence, flagged, corrected], [status])

if __name__ == "__main__":
    demo.launch()
