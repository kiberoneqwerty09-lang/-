from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re
app = Flask(__name__)
sentiment_analyser = pipeline("sentiment-analysis")
generator = pipeline("text-generation", model="sberbank-ai/rugpt2medium_based_on_gpt2")
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt2medium_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt2medium_based_on_gpt2")
def extract_film_title(generated_text: str) -> str:
    if not generated_text:
        return ""
    m = re.search(r'[А-Яа-яЁё\w\s]{3,100}', generated_text)
    if m:
        return m.group(0).strip()
    first_line = generated_text.strip().splitlines()[0]
    return first_line[:100]
def generate_recommendation(mood: str) -> str:
    prompt = (f"Посоветуй только один популярный фильм для человека, у которого настроение «{mood}». "
              f"Напиши только название фильма, без описаний и комментариев.")
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs.input_ids,
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.7
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    generated_part = text[len(prompt):].strip()
    return generated_part
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendation = ""
    user_text = ""
    if request.method == "POST":
        user_text = request.form["message"]
        result = sentiment_analyser(user_text)
        label = result[0]["label"]
        if label == "POSITIVE":
            mood = "хорошее"
        elif label == "NEGATIVE":
            mood = "плохое"
        else:
            mood = "нейтральное"

        raw_recommendation = generate_recommendation(mood)
        film_title = extract_film_title(raw_recommendation)
        recommendation = f"Настроение: {mood}<br>Рекомендация: {film_title}"
    return render_template('index.html', recommendation=recommendation, user_text=user_text)

if __name__ == "__main__":
    app.run(debug=True)



