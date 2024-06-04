from django.db import models
from datetime import datetime

class SS(models.Model):
    date = models.DateField()
    close = models.FloatField()

def load_ss(start_date, end_date):
    import FinanceDataReader as fdr

    ss_data = fdr.DataReader('005930', start_date, end_date)

    for index, row in ss_data.iterrows():
        ss_price = SS(
            date = index,
            close = row['Close']
        )
        ss_price.save()

class KS(models.Model):
    date = models.DateField()
    close = models.FloatField()

def load_ks(start_date, end_date):
    import FinanceDataReader as fdr

    ks_data = fdr.DataReader('KS11', start_date, end_date)

    for index, row in ks_data.iterrows():
        ks_price = KS(
            date = index,
            close = row['Close']
        )
        ks_price.save()


class Prediction(models.Model):
    date = models.DateField() 
    actual = models.FloatField(null=True, blank=True)
    prediction = models.FloatField(null=True, blank=True)
    difference = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ['date']