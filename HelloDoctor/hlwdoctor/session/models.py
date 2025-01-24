from django.db import models

from django.contrib.auth.models import User
from django.db import models

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    age = models.IntegerField()
    phone_number = models.CharField(max_length=15)

    def __str__(self):
        return self.user.username