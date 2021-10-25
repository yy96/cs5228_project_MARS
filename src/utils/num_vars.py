import pandas as pd
import datetime
import numpy as np

def calculate_arf(omv):
  if omv <= 20000:
      arf = omv
  elif omv <= 50000:
      arf = 20000 + (omv-20000)*1.4
  elif omv >50000:
      arf = 20000 + 30000*1.4 + (omv-50000)*1.8
  elif pd.isnull(omv):
      arf = np.NaN
  return arf

def find_coe_date(title, is_coe_car):
  if is_coe_car == 1 and 'COE' in title:
      try:
          coe_date = title.split('(')[1].split(')')[0].split(' ')[2]
          return datetime.datetime.strptime(coe_date, "%m/%Y")
      except:
          return np.NaN
  return np.NaN

def calculate_dereg_value(arf, car_age, coe_month_left, coe_date, coe, is_coe_car):
  try:
      if is_coe_car == 1 and not pd.isnull(coe_date):
          # taking as of this 2021-10
          # unused coe duration * coe
          coe_val = coe* (coe_month_left/120)
      else:
          coe_val = 0

      if car_age <= 5:
          parf = 0.75 * arf
      elif car_age <= 6:
          parf = 0.7 * arf
      elif car_age <= 7:
          parf = 0.65 * arf
      elif car_age <= 8:
          parf = 0.6 * arf
      elif car_age <= 9:
          parf = 0.55 * arf
      elif car_age <= 10:
          parf = 0.5 * arf
      else:
          parf = 0
  
      return coe_val+parf
  except:
      return np.NaN

def calculate_depreciation(omv, arf, coe, dereg_value, car_age):
  try: 
      rf = 140
      return (omv+arf+coe+rf - dereg_value)/car_age
  except:
      return np.NaN

def calculate_road_tax(engine_cap, car_age):
  try:
      if engine_cap <= 600:
          tax = 400 * 0.782
      elif engine_cap <= 1000:
          tax = (400 + 0.25*(engine_cap - 600))*0.782
      elif engine_cap <= 1600:
          tax = (500 + 0.75*(engine_cap - 1000))*0.782
      elif engine_cap <=3000:
          tax = (950 + 1.5*(engine_cap - 1600))*0.782
      elif engine_cap > 3000: 
          tax = (3050 + 2*(engine_cap - 3000))*0.782
      
      if car_age <= 10 or pd.isnull(car_age):
          multiplier = 1
      elif car_age <= 11:
          multiplier = 1.1
      elif car_age <= 12:
          multiplier = 1.2
      elif car_age <= 13:
          multiplier = 1.3
      elif car_age <= 14:
          multiplier = 1.4
      else:
          multiplier = 1.5
      return tax * multiplier
  except:
      return np.NaN
      