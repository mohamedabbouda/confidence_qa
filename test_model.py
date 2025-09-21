from transformers import pipeline

# Load the pretrained model fine-tuned on SQuAD
qa_pipeline = pipeline(
    "question-answering",
    model="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
    tokenizer="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
)

# Example context and question
context = "The Eiffel Tower is in Paris. It was completed in 1889."
question = "Where is the Eiffel Tower?"

result = qa_pipeline(question=question, context=context)
print(result)
