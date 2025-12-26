def teindex(i, j, k, l):
    i, j = max(i, j), min(i, j)
    k, l = max(k, l), min(k, l)

    ij = i * (i + 1) // 2 + j
    kl = k * (k + 1) // 2 + l

    ij, kl = max(ij, kl), min(ij, kl)

    return ij * (ij + 1) // 2 + kl