from django.shortcuts import render
import torch  # Or import tensorflow as tf
from .models import Llm_model, Politician

def main_page_view(request, *args, **kwargs):
    queryset = Politician.objects.all()
    context = {
        "object_politicians": queryset
    }
    return render(request, "index.html", context)

def chat_view(request, politician_id):
    print(politician_id)
    politician = Politician.objects.get(pk=politician_id)
    queryset = Politician.objects.all()
    context = {
        "object_politicians": queryset,
        "object_chat": politician
    }

    # floors = Floor.objects.all()
    # floor = floors.get(id=room.floor.id)
    return render(request, 'chat_view.html', context)

# def compute_nn_output(request):
#     output = None

#     if request.method == "POST":
#         user_input = request.POST['input']
#         # Retrieve the latest NN parameters from the database
#         nn_params = NeuralNetwork.objects.latest('id').parameters
        
#         # Convert the parameters to your framework's format and compute the output
#         # Here, I'm simplifying it greatly.
#         model = load_model_from_params(nn_params)
#         output = model.forward(user_input)

#     return render(request, 'index.html', {'output': output})
