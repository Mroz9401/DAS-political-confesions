from django.shortcuts import render
import torch  # Or import tensorflow as tf
import json
from .models import Llm_model, Politician
from .llm_utils import get_llm_response
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

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

@require_POST
@csrf_exempt  # Only use this if you are not using CSRF tokens, otherwise handle CSRF properly
def llm_response_view(request):
    try:
        data = json.loads(request.body)
        message = data['message']
        object_chat = data['object_chat']
        # Now call your Python function that generates the LLM output
        politician = Politician.objects.get(pk=object_chat)
        llm_output = get_llm_response(message, politician.llm_polit.path)
        return JsonResponse({'llm_output': llm_output})
    except Exception as e:
        # Log the error here if you need to
        return JsonResponse({'error': str(e)}, status=500)


