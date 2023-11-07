from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def generate_text(model_path, sequence, max_length):

    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    # ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)

# Inside your llm module where get_llm_response is defined

def get_llm_response(message, object_chat):
    # Your logic to generate the LLM response goes here
    response = generate_text(object_chat, message, 50)
    return response


# def compute_llm_output(request):
#     output = None

#     if request.method == "POST":
#         model2_path = base_path + "/model"
#         sequence2 = "Question:"
#         max_len = 50
        
#         output = generate_text(model2_path, sequence2, max_len)

#     return render(request, 'index.html', {'output': output})