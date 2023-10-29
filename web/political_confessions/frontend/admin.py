from django.contrib import admin
from .models import Llm_model, Politician

# Register your models here.
admin.site.register(Llm_model)
admin.site.register(Politician)