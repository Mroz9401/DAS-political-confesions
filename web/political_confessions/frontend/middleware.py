from django.contrib import messages
from django.shortcuts import redirect
from django.urls import reverse
from urllib.parse import urlencode

class LoginRequiredMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        return response

    def process_view(self, request, view_func, view_args, view_kwargs):
        if not request.user.is_authenticated and request.path != reverse('login'):
            query_string = urlencode({'next': request.path})
            login_url = reverse('login') + '?' + query_string
            return redirect(login_url)
        return None