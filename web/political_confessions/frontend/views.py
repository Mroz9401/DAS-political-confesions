from django.shortcuts import render, redirect
import json
from .models import Embeddings, Politician
from .llm_utils import get_llm_response
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib import messages


def user_login_view(request):
    if request.GET.get('next'):
        messages.error(request, "You need to log in first!")
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('homepage')  # Redirect to a success page.
        else:
            return render(request, 'index.html', {'error': 'Invalid credentials'})
    else:
        return render(request, 'index.html')

@login_required
def homepage_view(request, *args, **kwargs):
    queryset = Politician.objects.all()
    context = {
        "object_politicians": queryset
    }
    return render(request, "homepage.html", context)

@login_required
def chat_view(request, politician_id):
    print(politician_id)
    politician = Politician.objects.get(pk=politician_id)
    queryset = Politician.objects.all()
    context = {
        "object_politicians": queryset,
        "object_chat": politician
    }

    return render(request, 'chat_view.html', context)

@login_required
@require_POST
@csrf_exempt  # Only use this if you are not using CSRF tokens, otherwise handle CSRF properly
def llm_response_view(request):
    try:
        data = json.loads(request.body)
        message = data['message']
        object_chat = data['object_chat']
        # Now call your Python function that generates the LLM output
        politician = Politician.objects.get(pk=object_chat)
        llm_output = get_llm_response(message, politician.llm_polit.key_path, politician.llm_polit.embedding_path, politician.llm_polit.model_name, politician.llm_polit.template_prompt)
        return JsonResponse({'llm_output': llm_output})
    except Exception as e:
        # Log the error here if you need to
        return JsonResponse({'error': str(e)}, status=500)


