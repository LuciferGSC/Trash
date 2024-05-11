from django.db import models

# Create your models here.
class Classifyinfo(models.Model):
    name = models.CharField(max_length=10)
    finalResult0 = models.CharField(max_length=1000)
    finalResult1 = models.CharField(max_length=1000)
    finalResult2 = models.CharField(max_length=1000)
    finalResult3 = models.CharField(max_length=1000)
    finalResult4 = models.CharField(max_length=1000)

