import openai

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def get_llm_response(input_message, model_path):
    api_key = open_file(model_path)
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model = "ft:gpt-3.5-turbo-1106:personal::8IUwlYFN",
        messages=[
            {"role": "system", "content": "Jsi Andrej Babiš, Český politik a podnikatel."},
            {"role": "user", "content": input_message}
        ] 
    )
    response = response.choices[0].message
    return response["content"]

