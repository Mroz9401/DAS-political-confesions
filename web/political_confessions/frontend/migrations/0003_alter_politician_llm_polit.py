# Generated by Django 4.2.6 on 2023-10-29 18:27

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('frontend', '0002_rename_neuralnetwork_llm_model_politician'),
    ]

    operations = [
        migrations.AlterField(
            model_name='politician',
            name='LLM_polit',
            field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='LLM', to='frontend.llm_model', verbose_name='llm_model'),
        ),
    ]