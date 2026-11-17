from itertools import combinations
import numpy as np

# ---------------- Álgebra lineal sobre F_2 ----------------
def fila_a_entero(fila):
    """Fila binaria -> entero (bit j = entrada j)."""
    return sum(1 << j for j, b in enumerate(fila) if b)

def base_reducida(M):
    """Base del espacio fila de M (eliminación gaussiana), indexada por bit principal."""
    base = {}
    for fila in M:
        v = fila_a_entero(fila)
        while v:
            b = v.bit_length() - 1          # bit principal de v
            if b not in base:
                base[b] = v
                break
            v ^= base[b]                     # suma módulo 2 (XOR)
    return base

def rango_f2(M):
    return len(base_reducida(M))

def base_nucleo(H):
    """Base del código C = {x : H x^T = 0}."""
    H = np.array(H, dtype=np.uint8) % 2
    m, n = H.shape
    A = H.copy()
    pivotes, r = [], 0
    for c in range(n):
        filas = np.nonzero(A[r:, c])[0]
        if filas.size == 0:
            continue
        p = r + filas[0]
        A[[r, p]] = A[[p, r]]
        for i in np.nonzero(A[:, c])[0]:
            if i != r:
                A[i] ^= A[r]
        pivotes.append(c)
        r += 1
        if r == m:
            break
    libres = [c for c in range(n) if c not in pivotes]
    base = []
    for f in libres:
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1
        for i, c in enumerate(pivotes):
            x[c] = A[i, f]
        base.append(x)
    return base

# ---------------- Matrices de paridad ----------------
def matriz_hamming(r):
    """H_r: columnas = 1, ..., 2^r - 1 en binario."""
    n = 2 ** r - 1
    return np.array([[(c >> (r - 1 - i)) & 1 for c in range(1, n + 1)]
                     for i in range(r)], dtype=np.uint8)

def permutacion_circulante(p, e):
    """P^e: unos en las posiciones (i, i+e mod p)."""
    P = np.zeros((p, p), dtype=np.uint8)
    for i in range(p):
        P[i, (i + e) % p] = 1
    return P

def matriz_qc(p):
    """H(p): bloque (a, b) = P^(ab mod p), a = 0,1,2 y b = 0,...,3."""
    return np.vstack([np.hstack([permutacion_circulante(p, (a * b) % p)
                                 for b in range(4)]) for a in range(3)])

# ---------------- Producto de hipergrafos ----------------
def producto_hipergrafos(H1, H2):
    """H_X = (H1 x I | I x H2^T),  H_Z = (I x H2 | H1^T x I)."""
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    I = lambda k: np.eye(k, dtype=np.uint8)
    HX = np.hstack([np.kron(H1, I(n2)), np.kron(I(m1), H2.T)]) % 2
    HZ = np.hstack([np.kron(I(n1), H2), np.kron(H1.T, I(m2))]) % 2
    return HX.astype(np.uint8), HZ.astype(np.uint8)

# ---------------- Distancia de un código lineal ----------------
def distancia_clasica(H, peso_max=12):
    """Menor peso de una palabra no nula de ker(H); None si el código es trivial."""
    H = np.array(H, dtype=np.uint8) % 2
    base = base_nucleo(H)
    k = len(base)
    if k == 0:
        return None
    if k <= 20:                              # recorrido de las 2^k palabras
        B = np.array(base, dtype=np.int64)
        mejor = H.shape[1]
        for coef in range(1, 2 ** k):
            elegidas = [i for i in range(k) if (coef >> i) & 1]
            peso = int((B[elegidas].sum(axis=0) % 2).sum())
            mejor = min(mejor, peso)
        return mejor
    columnas = [fila_a_entero(H[:, j]) for j in range(H.shape[1])]
    sumas = {0: {0}}
    def sumas_de(t):                         # sumas de t columnas distintas
        if t not in sumas:
            s = set()
            for comb in combinations(columnas, t):
                x = 0
                for c in comb:
                    x ^= c
                s.add(x)
            sumas[t] = s
        return sumas[t]
    for w in range(1, peso_max + 1):         # encuentro a mitad de camino
        a, b = (w + 1) // 2, w // 2
        if a == b:
            conteo = {}
            for comb in combinations(range(len(columnas)), a):
                x = 0
                for j in comb:
                    x ^= columnas[j]
                conteo[x] = conteo.get(x, 0) + 1
                if conteo[x] > 1:
                    return w
        elif sumas_de(a) & sumas_de(b):
            return w
    raise ValueError("distancia mayor que peso_max")

# ---------------- Parámetros del código HGP con H1 = H2 = H ----------------
def parametros_hgp(H):
    m, n = H.shape
    HX, HZ = producto_hipergrafos(H, H)
    assert not ((HX.astype(int) @ HZ.T.astype(int)) % 2).any()   # H_X H_Z^T = 0
    rH = rango_f2(H)
    k1, k1T = n - rH, m - rH
    N = HX.shape[1]
    K = N - rango_f2(HX) - rango_f2(HZ)
    assert K == k1 * k1 + k1T * k1T
    d1 = distancia_clasica(H)
    d1T = distancia_clasica(H.T) if k1T > 0 else None
    D = min(x for x in (d1, d1T) if x is not None)
    pesos = HX.sum(axis=1).tolist() + HZ.sum(axis=1).tolist()
    return dict(n1=n, m1=m, rango=rH, k1=k1, k1T=k1T, d1=d1, d1T=d1T,
                N=N, K=K, D=D, peso_min=min(pesos), peso_max=max(pesos),
                HX=HX, HZ=HZ)
