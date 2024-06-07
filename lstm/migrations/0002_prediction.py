# Generated by Django 5.0.6 on 2024-06-04 16:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('lstm', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Prediction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('actual', models.FloatField()),
                ('prediction', models.FloatField()),
            ],
        ),
    ]