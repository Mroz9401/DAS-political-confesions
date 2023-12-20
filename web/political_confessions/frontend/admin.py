from django.contrib import admin
from .models import Embeddings, Politician

# Register your models here.
admin.site.register(Embeddings)
admin.site.register(Politician)