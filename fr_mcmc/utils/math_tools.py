import math

def round_to_significant_digits(value, digits):
    if value == 0:
        return 0
    else:
        return round(value, digits - int(math.floor(math.log10(abs(value)))) - 1)

#%%
if __name__ == '__main__':
    # Example usage:
    value = 0.99916
    rounded_value = round_to_significant_digits(value, 3)
    print(rounded_value)  # Output: 0.999