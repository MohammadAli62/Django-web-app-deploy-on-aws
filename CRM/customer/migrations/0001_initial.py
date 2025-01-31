# Generated by Django 5.0.4 on 2024-04-05 11:44

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Invoice',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('invoice_no', models.IntegerField(max_length=100)),
                ('description', models.CharField(max_length=255)),
                ('quantity', models.IntegerField()),
                ('invoice_date', models.DateField()),
                ('unit_price', models.DecimalField(decimal_places=2, max_digits=10)),
                ('customer_id', models.IntegerField(max_length=100)),
                ('country', models.CharField(max_length=100)),
            ],
        ),
    ]
