# qa_pipeline.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datasets import load_dataset
from transformers import pipeline

# Pretrained extractive QA model fine-tuned on SQuAD
# Model card: https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad
MODEL_ID = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"

# Default confidence threshold (you can change this in the app)
CONF_THRESHOLD_DEFAULT = 0.75

@dataclass
class QAResult:
    context_id: str
    question: str
    predicted_answer: str
    confidence: float
    flagged: bool
    start: int
    end: int

class QAPredictor:
    def __init__(self, model_id: str = MODEL_ID, device: Optional[int] = None):
        # The "question-answering" pipeline returns {'answer', 'score', 'start', 'end'}.
        # We treat 'score' as the confidence.
        self.qa = pipeline("question-answering", model=model_id, tokenizer=model_id, device=device)

    def answer_one(self, question: str, context: str, threshold: float = CONF_THRESHOLD_DEFAULT) -> QAResult:
        if not question or not context:
            return QAResult("", question or "", "", 0.0, True, -1, -1)
        out: Dict[str, Any] = self.qa(question=question, context=context)
        predicted = out.get("answer", "").strip()
        conf = float(out.get("score", 0.0))
        start = int(out.get("start", -1))
        end = int(out.get("end", -1))
        flagged = conf < threshold
        return QAResult("", question, predicted, conf, flagged, start, end)

def run_sample_and_log_flags(
    sample_size: int = 50,
    split: str = "validation",
    threshold: float = CONF_THRESHOLD_DEFAULT,
    log_path: str = "feedback_log.jsonl",
):
    import json
    # SQuAD has fields: context, question, answers (with text and answer_start)
    ds = load_dataset("squad", split=split)
    ds = ds.select(range(min(sample_size, len(ds))))

    predictor = QAPredictor()

    with open(log_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            res = predictor.answer_one(row["question"], row["context"], threshold)
            res.context_id = f"doc_{i:04d}"
            if res.flagged:
                rec = {
                    "context_id": res.context_id,
                    "question": row["question"],
                    "context": row["context"],
                    "predicted_answer": res.predicted_answer,
                    "confidence": round(res.confidence, 4),
                    "flagged": True,
                    "ground_truth": row["answers"]["text"][0] if row["answers"]["text"] else ""
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    run_sample_and_log_flags()
