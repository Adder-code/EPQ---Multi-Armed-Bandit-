import random

numbers = list(range(1, 101))
missing = []
for _ in range(100):
    generated = [random.randint(1, 100) for _ in range(100)] 


    print("Generated numbers:")
    print(generated)


    missing_numbers = set(numbers) - set(generated)

    print(f"\n{len(missing_numbers)} numbers were not generated:")
    print(sorted(missing_numbers))  
    missing.append(len(missing_numbers))

avg_missing = sum(missing)/len(missing)

print(f'\nThe averaged missing generated was {avg_missing}')