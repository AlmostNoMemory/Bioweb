# Generated by Django 2.1 on 2019-07-28 15:54

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='IMG',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('img', models.ImageField(max_length=10000, upload_to='img')),
                ('name', models.CharField(max_length=50)),
            ],
        ),
    ]
