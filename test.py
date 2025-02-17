unique = []
for a in range(5):
    for b in range(5):
        for c in range(4):
            for d in range(4):
                unique.append(d+c*4+b*4*4+a*5*4*4)

print(len(unique) == len(set(unique)))