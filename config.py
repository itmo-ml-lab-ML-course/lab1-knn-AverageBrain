import re

ACCEPTABLE_PERCENT_OF_NULL = 0.25
ID = lambda x: x
TO_INT_NUMBERS = (lambda x: int(re.sub(r'\D', '', x)))
TO_FLOAT_NUMBERS = (lambda x: float(re.sub(r'[^\d.]', '', x.replace(',', '.'))))


COLUMNS = {'price': TO_INT_NUMBERS,
           'year': TO_INT_NUMBERS,
           'origin': ID,
           'firstCirculationDate': ID,
           'technicalControl': ID,
           'firstHand': ID,
           'mileage': TO_INT_NUMBERS,
           'energy': ID,
           'gearbox': ID,
           'externalColor': ID,
           'doors': TO_INT_NUMBERS,
           'seats': TO_INT_NUMBERS,
           'length': TO_FLOAT_NUMBERS,
           'trunkVolumeRange': ID,
           'consumption': TO_FLOAT_NUMBERS,
           'co2': TO_INT_NUMBERS
        }

