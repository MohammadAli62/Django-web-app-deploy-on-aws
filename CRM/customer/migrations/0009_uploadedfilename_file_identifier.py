# Generated by Django 5.0.4 on 2024-12-01 13:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('customer', '0008_uploadedfilename'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadedfilename',
            name='file_identifier',
            field=models.IntegerField(default=1),
        ),
    ]
