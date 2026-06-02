# Máximo Paralelo com Árvore Binária Balanceada e OpenMP

Implementação em C do algoritmo de **elemento máximo de um vetor** usando a estrutura de **árvore binária balanceada** paralelizada com **OpenMP**, nas duas versões exigidas pelo **Teorema de Brent**.

---

## Descrição do Problema

Dado um vetor de `N` inteiros, encontrar o **elemento máximo** de forma paralela, utilizando a estrutura de árvore binária balanceada e o modelo de custo do Teorema de Brent para justificar a eficiência da paralelização.

---

## Conceito: Árvore Binária Balanceada

A ideia central é tratar a busca pelo máximo como um **torneio eliminatório**. Os elementos são comparados em pares a cada rodada, e apenas o maior "sobe" para a próxima. A estrutura resultante é uma árvore binária completa.

```
Rodada 0 (folhas): [ 3, 7, 2, 9, 5, 1, 8, 4 ]
                      |   |   |   |
Rodada 1:          [  7,    9,   5,   8 ]
                      |        |
Rodada 2:          [     9,       8   ]
                          |
Rodada 3 (raiz):   [      9           ]  ← máximo
```

Com `N` elementos:

| Rodada | Comparações |
|--------|-------------|
| 1      | N/2         |
| 2      | N/4         |
| ...    | ...         |
| log₂N  | 1           |
| **Total** | **N − 1** |

- **Profundidade (passos críticos):** `O(log N)`
- **Trabalho total:** `O(N)`

---

## Teorema de Brent

O Teorema de Brent fornece um limite superior para o tempo de execução de um algoritmo paralelo com `p` processadores:

```
T_p  <=  T_∞  +  (T_1 − T_∞) / p
```

Onde:

| Símbolo | Significado |
|---------|-------------|
| `T_p`   | Tempo com `p` processadores |
| `T_∞`   | Tempo com processadores infinitos (caminho crítico) |
| `T_1`   | Tempo sequencial (trabalho total) |
| `p`     | Número de processadores |

Aplicado a este problema com `N = 2²⁰`:

```
T_∞ = log₂(N) = 20  passos críticos
T_1 = N        = 1.048.576  comparações

T_p  <=  20  +  (1.048.576 − 20) / p
```

| p | Limite superior (comparações) |
|---|-------------------------------|
| 1 | 1.048.576                     |
| 2 | 524.298                       |
| 4 | 262.159                       |
| 8 | 131.090                       |

---

## Versão 1 — Árvore por Rodadas (Paralela Explícita)

### Estratégia

Implementa a árvore **manualmente**, nível por nível. Cada iteração do laço principal corresponde a uma **rodada do torneio**. As comparações dentro de cada rodada são **independentes entre si**, portanto paralelizadas com `#pragma omp parallel for`.

Para evitar conflito de leitura/escrita entre threads na mesma rodada, são usados **dois buffers alternados**: sempre lemos de `src[]` e escrevemos em `dst[]`, trocando os papéis ao final de cada rodada.

### Pseudocódigo

```
buf_a[] ← cópia do vetor original
src = buf_a,  dst = buf_b
tamanho = N

enquanto tamanho > 1:
    metade = tamanho / 2

    PARALELO para i = 0 até metade-1:
        dst[i] = max(src[2i], src[2i+1])

    se tamanho é ímpar:
        dst[metade] = src[tamanho-1]
        tamanho = metade + 1
    senão:
        tamanho = metade

    troca src ↔ dst

retorna src[0]
```

### Diretiva OpenMP utilizada

```c
#pragma omp parallel for num_threads(p) schedule(static)
for (int i = 0; i < metade; i++) {
    dst[i] = (src[2*i] > src[2*i+1]) ? src[2*i] : src[2*i+1];
}
```

### Complexidade (Brent)

```
T_p  <=  log₂(N)  +  (N − log₂(N)) / p
```

---

## Versão 2 — Redução OpenMP (`reduction` clause)

### Estratégia

Delega ao OpenMP a tarefa de organizar a árvore binária. A cláusula `reduction(max:)` faz com que cada thread calcule o **máximo local** da sua fatia do vetor e, ao final, o runtime do OpenMP combina todos os máximos locais usando internamente uma **árvore de redução em `log(p)` passos**.

### Como o OpenMP implementa a redução

1. Cria uma **cópia privada** de `maximo` para cada thread, inicializada com `INT_MIN`
2. Cada thread percorre sua fatia e atualiza seu máximo local (fase paralela: `N/p` comparações por thread)
3. Ao final da região paralela, combina os `p` máximos locais em árvore binária (`log(p)` passos)

### Diretiva OpenMP utilizada

```c
#pragma omp parallel for num_threads(p) reduction(max : maximo) schedule(static)
for (int i = 0; i < n; i++) {
    if (vet[i] > maximo) maximo = vet[i];
}
```

### Complexidade (Brent implícito)

```
T_p  ≈  N/p  +  log(p)
```

---

## Estrutura do Projeto

```
.
├── maximo_openmp.c   ← código-fonte principal
└── README.md         ← este arquivo
```

---

## Como Compilar

```bash
gcc -O2 -fopenmp -o maximo_openmp maximo_openmp.c
```

| Flag        | Descrição |
|-------------|-----------|
| `-O2`       | Otimizações de desempenho |
| `-fopenmp`  | Habilita suporte a OpenMP |
| `-o maximo_openmp` | Nome do executável gerado |

---

## Como Executar

```bash
./maximo_openmp
```

O programa executa automaticamente as duas versões com `p = 1, 2, 4, 8` threads e imprime os resultados, tempo de execução e speedup de cada configuração.

Para controlar o número máximo de threads via variável de ambiente (opcional):

```bash
OMP_NUM_THREADS=4 ./maximo_openmp
```

---

## Saída Esperada

```
=======================================================
  Maximo com Arvore Binaria Balanceada + OpenMP
  Teorema de Brent -- Duas Versoes
=======================================================
  Tamanho do vetor : N = 1048576 (2^20)
  Maximo correto   : 999999
=======================================================

--- VERSAO 1: Arvore por Rodadas (paralela explicita) ---
  Complexidade: T_p <= log2(N) + (N - log2(N)) / p

  p =  1 | Maximo =  999999 | Tempo = 0.02584 s | Speedup = 1.00x | OK
  p =  2 | Maximo =  999999 | Tempo = 0.00378 s | Speedup = 6.83x | OK
  p =  4 | Maximo =  999999 | Tempo = 0.00218 s | Speedup = 11.83x | OK
  p =  8 | Maximo =  999999 | Tempo = 0.00273 s | Speedup = 9.46x | OK

--- VERSAO 2: Reducao OpenMP (reduction clause) --------
  Complexidade: T_p aprox N/p + log(p)  (Brent implicito)

  p =  1 | Maximo =  999999 | Tempo = 0.00066 s | Speedup = 1.00x | OK
  p =  2 | Maximo =  999999 | Tempo = 0.00099 s | Speedup = 0.67x | OK
  p =  4 | Maximo =  999999 | Tempo = 0.00073 s | Speedup = 0.90x | OK
  p =  8 | Maximo =  999999 | Tempo = 0.00140 s | Speedup = 0.53x | OK

=======================================================
  Previsao de Brent para N=1048576:
  T_infinito (passos criticos) = log2(1048576) = 20 passos
  T_1 (trabalho total)         = 1048576 comparacoes
  T_p <= 20 + (1048576 - 20) / p
    p=1 -> T_p <= 1048576 comparacoes logicas
    p=2 -> T_p <= 524298 comparacoes logicas
    p=4 -> T_p <= 262159 comparacoes logicas
    p=8 -> T_p <= 131090 comparacoes logicas
=======================================================
```

---

## Análise de Complexidade

### Versão 1

| Métrica | Valor |
|---------|-------|
| Trabalho total `T_1` | `O(N)` |
| Caminho crítico `T_∞` | `O(log N)` |
| Tempo com `p` threads | `T_p ≤ log₂N + (N − log₂N) / p` |
| Eficiência paralela | `E = T_1 / (p · T_p)` |
| Memória auxiliar | `2 × N` inteiros (dois buffers) |

### Versão 2

| Métrica | Valor |
|---------|-------|
| Trabalho total `T_1` | `O(N)` |
| Caminho crítico `T_∞` | `O(log p)` |
| Tempo com `p` threads | `T_p ≈ N/p + log(p)` |
| Eficiência paralela | Alta para `N >> p` |
| Memória auxiliar | `O(p)` variáveis privadas (gerenciadas pelo OpenMP) |

---

## Comparação entre as Versões

| Critério | Versão 1 (Rodadas) | Versão 2 (Reduction) |
|----------|-------------------|----------------------|
| Controle da paralelização | Explícito (manual) | Implícito (OpenMP) |
| Estrutura da árvore | Visível no código | Interna ao runtime |
| Sincronização entre rodadas | Barreira implícita por nível | Uma única barreira ao final |
| Overhead de paralelismo | Alto (muitas regiões paralelas) | Baixo (uma única região) |
| Uso de memória extra | `2N` inteiros | `p` variáveis |
| Portabilidade do conceito | Alta (mostra o algoritmo) | Alta (idiomático em OpenMP) |
| Melhor para fins didáticos | ✅ Sim | ✅ Sim (mostra abstração) |
| Melhor desempenho prático | Depende do hardware | Geralmente mais rápido |

A **Versão 1** é mais adequada para demonstrar explicitamente o funcionamento da árvore binária e a aplicação do Teorema de Brent. A **Versão 2** é mais idiomática e eficiente na prática, pois evita o overhead de criar e destruir múltiplas regiões paralelas a cada rodada.