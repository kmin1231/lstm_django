# Generated by Django 5.0.6 on 2024-06-04 21:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('lstm', '0005_alter_prediction_actual_alter_prediction_date_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='prediction',
            name='prediction',
            field=models.FloatField(null=True),
        ),
    ]
