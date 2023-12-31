# Generated by Django 4.2.6 on 2023-10-29 20:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('frontend', '0004_llm_model_name_alter_politician_llm_polit'),
    ]

    operations = [
        migrations.RenameField(
            model_name='politician',
            old_name='LLM_polit',
            new_name='llm_polit',
        ),
        migrations.AlterField(
            model_name='politician',
            name='icon_path',
            field=models.ImageField(default='img/babis.jpg', upload_to='frontend/img'),
        ),
    ]
