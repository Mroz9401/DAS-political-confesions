from django.db import models

class Embeddings(models.Model):
    name = models.TextField()
    model_name = models.CharField(max_length=1024, blank=True)
    embedding_path = models.CharField(max_length=1024, blank=True)
    key_path = models.CharField(max_length=1024, blank=True)
    template_prompt = models.TextField()

    def __str__(self):
        return self.name

class Politician(models.Model):
    name = models.TextField()  # You can use a more sophisticated method to store parameters.
    icon_path = models.ImageField(upload_to='frontend/img',  default='img/babis.jpg')
    llm_polit = models.ForeignKey(
        Embeddings,
        on_delete=models.PROTECT,
        verbose_name=('Embeddings'),
        related_name='LLM'
    )

    def __str__(self):
        return self.name

