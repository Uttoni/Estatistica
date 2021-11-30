import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

if __name__ == '__main__':

    #Gerando X e Y
    mediaX = 5
    desvioPadraoX = 1
    x = np.random.normal(mediaX, desvioPadraoX, 10000)
    mediaY = 6
    desvioPadraoY = 1
    y = np.random.normal(mediaY, desvioPadraoY, 10000)

    #1) Histograma de X
    plt.title("Histograma de X")
    plt.hist(x, 100)
    plt.show()

    #2) Histograma de Y
    plt.title("Histograma de Y")
    plt.hist(y, 100)
    plt.show()

    #3) Distribuição cumulativa X
    plt.title("FDC de X")
    plt.hist(x, cumulative=True, density=True, bins=100)
    plt.show()

    #4) Distribuição cumulativa Y
    plt.title("FDC de Y")
    plt.hist(y, cumulative=True, density=True, bins=100)
    plt.show()

    #5) Probabilidade calculada de x>6, P[x>6]
    print("")
    print("#5------------")
    print("Calculado P[x>6]=", stats.norm.sf(x=6, loc=mediaX, scale=desvioPadraoX)*100, "%")

    # 5) Probabilidade calculada de x>6, P[x>6]
    cont = 0
    for i in x:
        if(i>6):
            cont = cont+1

    print("")
    print("#6------------")
    print("Estimado P[x>6]=", (cont*100/10000), "%")


    #7)
    print("")
    print("#7------------")
    print("Em uma distribuição gaussiana a probabilidade da variável ser exatamente um número é 0.")

    #8) P[y=0] = P[y]<0.0001 - p[y]<0
    print("")
    print("#8------------")
    print("P[y=0]=", ((stats.norm.sf(x=0.0001, loc=mediaY, scale=desvioPadraoY)-
                      stats.norm.sf(x=0, loc=mediaY, scale=desvioPadraoY))*100), "%")
    print("Aproximadamente 0")

    #9) Teste de Kolmogorov-Smirnov
    print("")
    print("#9-----------")
    # Verifica o valor critico com 95% de confiança
    def kolmogorov_smirnov_critico(n):
        kolmogorov_critico = 1.36 / (np.sqrt(n))
        ks_critico = kolmogorov_critico
        return ks_critico

    ks_critico = kolmogorov_smirnov_critico(len(x))

    # Calculando o valor de Kolmogorov-Smirnov
    ks_stat, ks_p_valor = stats.kstest(x, cdf='norm', args=(mediaX, desvioPadraoX), N=10000)

    print("Com 95% de confianca, o valor critico do teste de Kolmogorov-Smirnov = " + str(ks_critico))
    print("O valor calculado do teste de Kolmogorov-Smirnov eh de = " + str(ks_stat))

    if ks_critico >= ks_stat:
        print("Com 95% de confianca, os dados estão distribuidos em uma gaussiana")
    else:
        print("Com 95% de confianca, os dados não estão distribuidos em uma gaussiana")

    #10) comparação usando teste t Student
    print("")
    print("#10-----------")
    t_value, p_value = stats.ttest_ind(y, x)

    #5% de margem de erro
    if((p_value/2)<0.05):
        print("Com 95% de certeza X é maior que Y")
    else:
        print("Com 95% de certeza X é menor que Y")