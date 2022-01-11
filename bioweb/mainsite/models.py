from django.db import models


# Create your models here.

class IMG(models.Model):
    # 图片数据库模型
    img = models.ImageField(upload_to='img',max_length=10000)
    name = models.CharField(max_length=50)
