import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## starting some training bullshit
    so i need to finetune Qwen2.5-0.5B-Instruct...
    and the first step i guess is to load the model, with pretrained weights, and get it working with our harness

    and by harness: i mean its outputs of the FEN board move should update the board state
    """)
    return


@app.cell
def _():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    qwen_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_id)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwen_model_id, dtype="auto", device_map="auto"
    )
    return qwen_model, qwen_tokenizer


@app.cell
def _(qwen_model, qwen_tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Whats ur name."},
    ]
    prompt_text = qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = qwen_tokenizer(prompt_text, return_tensors="pt").to(qwen_model.device)

    generated_ids = qwen_model.generate(
        **model_inputs,
        max_new_tokens=128,
        do_sample=False,
    )
    new_tokens = generated_ids[0, model_inputs.input_ids.shape[1]:]
    response = qwen_tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(response)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
