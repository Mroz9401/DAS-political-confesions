import openai

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def get_llm_response(input_message, model_path, model_name, model_context):
    api_key = open_file(model_path)
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model = model_name,
        messages=[
            {"role": "system", "content": model_context},
            {"role": "assistant", "content": input_message}
        ] 
    )
    response = response.choices[0].message
    return response["content"]

