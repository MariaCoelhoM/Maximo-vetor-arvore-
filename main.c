/*
 * Algoritmo de Máximo com Árvore Binária Balanceada e OpenMP
 * Teorema de Brent — Duas Versões
 *
 * Compilar com:
 *   gcc -O2 -fopenmp -o maximo_openmp maximo_openmp.c
 *
 * Executar:
 *   ./maximo_openmp
 *
 * -------------------------------------------------------
 * CONCEITO: Árvore Binária Balanceada
 * -------------------------------------------------------
 * A ideia é comparar os elementos em "rodadas", como num
 * torneio. Em cada rodada, pares de elementos são comparados
 * e apenas o maior "sobe". Com N elementos:
 *   - Rodada 1: N/2 comparações
 *   - Rodada 2: N/4 comparações
 *   - ...
 *   - Total: O(log N) rodadas, O(N) trabalho
 *
 * -------------------------------------------------------
 * TEOREMA DE BRENT
 * -------------------------------------------------------
 * Com p processadores e T_infinito passos críticos:
 *   T_p <= T_infinito + (T_1 - T_infinito) / p
 *
 * Para N elementos e p processadores:
 *   T_infinito = O(log N)   (profundidade da árvore)
 *   T_1        = O(N)       (trabalho total)
 *   T_p        <= log(N) + (N - log(N)) / p
 *
 * -------------------------------------------------------
 * VERSÃO 1 — Paralela por Nível (top-down por rodadas)
 * -------------------------------------------------------
 * Cada rodada da árvore é paralelizada com OpenMP.
 * Usamos dois buffers alternados para evitar conflito de
 * leitura/escrita: enquanto lemos de "src", escrevemos em "dst".
 *
 * -------------------------------------------------------
 * VERSÃO 2 — Redução OpenMP (reduction clause)
 * -------------------------------------------------------
 * Usa a diretiva `reduction(max:)` do OpenMP, que
 * internamente implementa a mesma lógica de árvore binária
 * de forma otimizada, aplicando o Teorema de Brent
 * automaticamente ao dividir o trabalho entre threads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <omp.h>

/* ============================================================
 * VERSÃO 1 — Árvore Binária por Rodadas (Paralela Explícita)
 * ============================================================
 * Aplica o Teorema de Brent explicitamente:
 *   - Cada nível da árvore = 1 rodada de comparações
 *   - As comparações dentro de cada rodada são paralelas
 *   - Dois buffers alternados (src/dst) evitam conflito de dados
 *   - Brent garante T_p <= log(N) + (N - log(N)) / p
 */
int maximo_arvore_v1(int *vet, int n, int p) {
    /* Dois buffers: lemos de src[], escrevemos em dst[] */
    int *buf_a = (int *)malloc(n * sizeof(int));
    int *buf_b = (int *)malloc(n * sizeof(int));

    /* Inicializa src com o vetor original */
    for (int i = 0; i < n; i++) buf_a[i] = vet[i];

    int *src = buf_a;
    int *dst = buf_b;
    int tamanho = n;

    /*
     * Cada iteração = um nível da árvore binária
     * (equivale a uma "rodada" do torneio)
     *
     * Lemos sempre de src[] e gravamos em dst[],
     * sem conflito entre threads.
     */
    while (tamanho > 1) {
        int metade = tamanho / 2;

        /*
         * Paraleliza as comparações desta rodada.
         * Brent: o trabalho desta rodada é "metade" comparações,
         * distribuídas entre p threads → cada thread faz ~metade/p
         */
        #pragma omp parallel for num_threads(p) schedule(static)
        for (int i = 0; i < metade; i++) {
            /* Lê de src[], escreve em dst[] — sem conflito */
            dst[i] = (src[2 * i] > src[2 * i + 1])
                     ? src[2 * i]
                     : src[2 * i + 1];
        }

        /* Se o tamanho era ímpar, o último elemento "sobe" sem par */
        if (tamanho % 2 != 0) {
            dst[metade] = src[tamanho - 1];
            tamanho = metade + 1;
        } else {
            tamanho = metade;
        }

        /* Troca os papéis dos buffers para a próxima rodada */
        int *tmp = src;
        src = dst;
        dst = tmp;
    }

    int resultado = src[0];
    free(buf_a);
    free(buf_b);
    return resultado;
}

/* ============================================================
 * VERSÃO 2 — Redução OpenMP (reduction clause)
 * ============================================================
 * O OpenMP implementa internamente a árvore binária ao usar
 * reduction(max:). Cada thread calcula o máximo local de sua
 * fatia do vetor (fase paralela), depois combina os máximos
 * locais em árvore (fase de redução em log(p) passos).
 *
 * Pela análise de Brent:
 *   - Fase paralela: cada thread faz ~N/p comparações
 *   - Fase de redução: log(p) comparações sequenciais
 *   - Total: T_p = N/p + log(p) ≈ Brent para esse modelo
 */
int maximo_arvore_v2(int *vet, int n, int p) {
    int maximo = INT_MIN;

    /*
     * reduction(max:maximo) faz o OpenMP:
     *   1. Criar uma cópia privada de "maximo" para cada thread
     *   2. Cada thread processa sua fatia e atualiza seu máximo local
     *   3. Ao final, combina todos os máximos locais em árvore binária
     */
    #pragma omp parallel for num_threads(p) reduction(max : maximo) schedule(static)
    for (int i = 0; i < n; i++) {
        if (vet[i] > maximo)
            maximo = vet[i];
    }

    return maximo;
}

/* ============================================================
 * Função auxiliar: máximo sequencial (para validação)
 * ============================================================ */
int maximo_sequencial(int *vet, int n) {
    int m = vet[0];
    for (int i = 1; i < n; i++)
        if (vet[i] > m) m = vet[i];
    return m;
}

/* ============================================================
 * MAIN — Testes e análise de desempenho
 * ============================================================ */
int main(void) {
    /* Configurações do experimento */
    int n = 1 << 20;   /* N = 2^20 = ~1 milhão de elementos */
    int p_lista[] = {1, 2, 4, 8};
    int num_configs = sizeof(p_lista) / sizeof(p_lista[0]);

    /* Gera vetor aleatório */
    int *vet = (int *)malloc(n * sizeof(int));
    srand(42);
    for (int i = 0; i < n; i++)
        vet[i] = rand() % 1000000;

    /* Referência sequencial */
    int ref = maximo_sequencial(vet, n);

    printf("=======================================================\n");
    printf("  Maximo com Arvore Binaria Balanceada + OpenMP\n");
    printf("  Teorema de Brent -- Duas Versoes\n");
    printf("=======================================================\n");
    printf("  Tamanho do vetor : N = %d (2^20)\n", n);
    printf("  Maximo correto   : %d\n", ref);
    printf("=======================================================\n\n");

    /* ---- Versão 1 ---- */
    printf("--- VERSAO 1: Arvore por Rodadas (paralela explicita) ---\n");
    printf("  Complexidade: T_p <= log2(N) + (N - log2(N)) / p\n\n");

    double t1_seq = 0.0;
    for (int c = 0; c < num_configs; c++) {
        int p = p_lista[c];
        double t0 = omp_get_wtime();
        int res = maximo_arvore_v1(vet, n, p);
        double t1 = omp_get_wtime() - t0;

        if (p == 1) t1_seq = t1;

        printf("  p = %2d | Maximo = %7d | Tempo = %.5f s | "
               "Speedup = %.2fx | %s\n",
               p, res, t1,
               (p == 1) ? 1.0 : t1_seq / t1,
               (res == ref) ? "OK" : "ERRO");
    }

    /* ---- Versão 2 ---- */
    printf("\n--- VERSAO 2: Reducao OpenMP (reduction clause) --------\n");
    printf("  Complexidade: T_p aprox N/p + log(p)  (Brent implicito)\n\n");

    double t2_seq = 0.0;
    for (int c = 0; c < num_configs; c++) {
        int p = p_lista[c];
        double t0 = omp_get_wtime();
        int res = maximo_arvore_v2(vet, n, p);
        double t1 = omp_get_wtime() - t0;

        if (p == 1) t2_seq = t1;

        printf("  p = %2d | Maximo = %7d | Tempo = %.5f s | "
               "Speedup = %.2fx | %s\n",
               p, res, t1,
               (p == 1) ? 1.0 : t2_seq / t1,
               (res == ref) ? "OK" : "ERRO");
    }

    printf("\n=======================================================\n");
    printf("  Previsao de Brent para N=%d:\n", n);
    printf("  T_infinito (passos criticos) = log2(%d) = 20 passos\n", n);
    printf("  T_1 (trabalho total)         = %d comparacoes\n", n);
    printf("  T_p <= 20 + (%d - 20) / p\n", n);
    for (int c = 0; c < num_configs; c++) {
        int p = p_lista[c];
        double tp = 20.0 + (double)(n - 20) / p;
        printf("    p=%d -> T_p <= %.0f comparacoes logicas\n", p, tp);
    }
    printf("=======================================================\n");

    free(vet);
    return 0;
}