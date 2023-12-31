# Generated by Django 4.2.6 on 2023-11-27 23:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('frontend', '0006_llm_model_path'),
    ]

    operations = [
        migrations.RenameField(
            model_name='llm_model',
            old_name='path',
            new_name='key_path',
        ),
        migrations.RemoveField(
            model_name='llm_model',
            name='description',
        ),
        migrations.RemoveField(
            model_name='llm_model',
            name='parameters',
        ),
        migrations.AddField(
            model_name='llm_model',
            name='model_name',
            field=models.CharField(blank=True, max_length=1024),
        ),
    ]
