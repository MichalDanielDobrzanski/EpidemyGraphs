"""

    Program do modelowania epidemii (modelowanych za pomocą modelu SIS)
    na grafach losowych wygenerowanych za pomocą modelu Barabasi-Albert
    
    Instalowanie biblioteki do rysowania grafów:
        sudo pip3 install networkx
    
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from random import randint, uniform


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():
    # generowanie grafu - parametry
    total_nodes = 3400  # calkowita liczba wierzcholkow
    m_0 = 3  # liczba wierzcholkow w poczatkowym grafie pelnym
    m = 3  # liczba polaczen tworzonych z istniejacym grafem przez kazdy nowy wierzcholek, warunek: m <= m_0
    t = total_nodes-m_0  # liczba wierzcholkow do dodania (krokow generowania grafu)
    n_graphs = 5

    # parametry modelu SIS
    beta = 0.04  # zakazanie
    gamma = 0.03  # zdrowienie
    t_max = 200
    # n_sim = 25
    max_plots = 4

    graph = generate_graph(m_0, m, t)  # generowanie grafu
    print_graph(graph)  # wypisanie grafu w konsoli
    degrees = get_graph_degree(graph, True)

    plot_graph(graph, graph_layout='spring')  # rysowanie grafu - czasochlonne
    plot_graph_degree(graph, m_0, m, t, n_graphs)  # rysowanie wykresu z rozkladem stopni wierzcholkow grafu graph

    # return

    # celowe zakazenie:
    # infect(graph, randint(0, total_nodes))  # losowy wezel
    sick_node = 64
    infect(graph, sick_node)  # losowy wezel

    # policz i_k dla grafu:
    infected_vect = calc_i_k(graph)
    print(BColors.WARNING + 'poczatkowy stan sieci: ' + BColors.ENDC + 'infected_vect=', infected_vect)

    # policz gamma c:
    lambda_c_sim, lambda_c_the = calc_lambda_c(graph, degrees, infected_vect, sick_node)
    print(BColors.OKBLUE + '\t lambda_c_sim = ' + str(lambda_c_sim) + ',\n\t lambda_c_the = ' + str(lambda_c_the) +
          ",\n\t lambda = " + str(beta/gamma) + BColors.ENDC)

    print(BColors.BOLD + 't_max = ' + str(t_max) + BColors.ENDC)

    # wylicz rownania rozniczkowe:
    infected_vect_calc = propagate_sis(infected_vect, beta, gamma, t_max, degrees)
    print(BColors.HEADER + 'teoria: ' + BColors.ENDC + 'infected_vect_t_max=', infected_vect_calc[t_max - 1])

    # rozwiaz rownanie rozniczkowe w t=niesk.:
    infected_vect_inf = infinite_sis(infected_vect, beta, gamma, degrees)
    print(BColors.OKGREEN + 'teoria(w niesk.): ' + BColors.ENDC + 'infected_vect_inf=', infected_vect_inf)

    # przeprowadz symulacje:
    infected_vect_sim = simulate_sis(graph, beta, gamma, t_max, degrees)
    print(BColors.OKBLUE + 'symulacja: ' + BColors.ENDC + 'infected_vect_t_max=', infected_vect_sim[t_max - 1])

    # przeprowadz symulacje 2:
    # infected_vect_sim_av = list(simulate_average_sis(graph, beta, gamma, t_max, degrees, n_sim))
    # print(BColors.OKBLUE + (r'symulacja 2 (średnia z %d symulacji): ' % n_sim) + BColors.ENDC +
    #       'infected_vect_t_max=', infected_vect_sim_av[t_max - 1])

    d = 0
    for i in degrees:
        if i != 0 and max_plots > 0:
            plot_sis(t_max, infected_vect_calc, infected_vect_sim, beta, gamma, d)  # dla wezla o stopniu 4
            # plot_sis(t_max, infected_vect_calc, infected_vect_sim_av, beta, gamma, d)  # dla wezla o stopniu 4
            max_plots -= 1
        d += 1

    return


"""
    m_0 - liczba wierzcholkow w poczatkowym grafie pelnym
    m - ile polaczen ma utworzyc nowo dodany wierzcholek
    t - zadana chwila czasowa
"""


def generate_graph(m_0, m, t):
    print("Generowanie grafu dla: \n\tm_0 =", m_0, "\n\tm = ", m, "\n\tt =", t)

    # stworz wstepny graf (graf pełny - całkowicie połączony klaster węzłów):
    graph = dict()
    for m_i in range(0, m_0):
        l = [x+1 for x in range(m_0)]  # dodaj wszystkie wezly (indeksowanie od 1)
        del l[m_i]  # usun siebie
        graph[m_i + 1] = [0, l]  # zrob liste
    # zapelniaj go:
    for t_i in range(0, t):
        pref_addition(graph, m)  # dodaj nowy wezel, ktory stworzy m polaczen (na koniec slownika)
    return graph


def pref_addition(g, m):
    num_node = len(g) + 1
    # wybierz wezly, do ktorych polaczy sie nowy wezel
    total_p = 0
    for v in g.values():
        total_p += len(v)
    marked = []
    while len(marked) < m:
        p = randint(0, total_p)
        cum_p = 0
        for k, v in g.items():
            cum_p += len(v)
            if (k not in marked) and (p <= cum_p):
                marked.append(k)
                # updatuj graf (do wczesniejszego wezla dodaj wskazanie na nowy tworzony):
                g[k][1].append(num_node)
                break
    # updatuj graf (dodaj nowy wezel wraz z jego polaczeniami):
    new_node = {num_node: [0, marked]}
    g.update(new_node)
    return


def get_graph_degree(g, print_deg=False):
    # pobieram rozklad stopni wierzczholkow
    max_degree = 0
    for k, v in g.items():
        degree = len(v[1])
        if degree > max_degree:
            max_degree = degree

    degrees = [0 for _ in range(max_degree + 1)]
    for k, v in g.items():
        degree = len(v[1])
        degrees[degree] += 1

    if print_deg:
        print('Rozklad stopni wierzcholkow grafu:')
        print(degrees)

    return degrees


# Utilities:
def print_graph(g):
    print("Graf o", len(g), "wierzcholkach:")
    for k, v in g.items():
        print(k, ':', v)
    return


def plot_graph(g, graph_layout='shell', node_size=200, node_color='blue',
               node_alpha=0.3,  node_text_size=12, edge_color='blue',
               edge_alpha=0.3, edge_tickness=1, text_font='sans-serif'):
    #  labels=None, edge_text_pos=0.3,

    plt.figure(1)
    plt.title('Graficzne przedstawienie grafu')

    g2 = nx.Graph()

    for k, v in g.items():
        g2.add_node(k)

    for k, v in g.items():
        for i in range(0, len(v[1])):
            g2.add_edge(k, v[1][i])

    if graph_layout == 'spring':
        graph_pos = nx.spring_layout(g2)
    elif graph_layout == 'spectral':
        graph_pos = nx.spectral_layout(g2)
    elif graph_layout == 'random':
        graph_pos = nx.random_layout(g2)
    else:
        graph_pos = nx.shell_layout(g2)
        
    # draw
    nx.draw_networkx_nodes(g2, graph_pos, node_size=node_size,
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(g2, graph_pos, width=edge_tickness,
                           alpha=edge_alpha, edge_color=edge_color)
    nx.draw_networkx_labels(g2, graph_pos, font_size=node_text_size,
                            font_family=text_font)
           
    plt.show()
    return

'''
usrednianie rozkladu stopnia wierzcholka na podstawie kilku grafow
'''


def average_graph_degree(degrees, n_graphs, m_0, m, t):
    if n_graphs > 1:
        for i in range(n_graphs):
            print("Graf numer: ", i)
            gr = generate_graph(m_0, m, t)
            degs = get_graph_degree(gr)
            # zamien rozklad na prawdopodobienstwa
            degs = [x / sum(degs) for x in degs]
            print('degs', degs)
            # wybierz wiekszy rozklad
            # l_degs = s_degs = []
            if len(degs) > len(degrees):
                l_degs = degs
                s_degs = degrees
            else:
                l_degs = degrees
                s_degs = degs

            # iteruj po mniejszym
            for j in range(len(s_degs)):
                l_degs[j] += s_degs[j]

            degrees = l_degs

        degrees = [x / n_graphs for x in degrees]

    return degrees

'''
graphs - usrednianie rozkładu dla takiej ilosci grafow
alpha - parametr dla rozkladu potegowego dla sieci bezskalowych
ref_length - jak duze wzgledem rozkladu wierzcholkow P(k) maja byc dwa rozklady odniesienia
'''


def plot_graph_degree(g, m_0, m, t, n_graphs=5, alpha=3, ref_length=0.8):
    
    plt.figure(2)
    plt.title(r'Rozkład stopni wierzchołków P(k) w sieci BA dla chwili $t=%d$' % t)
    plt.ylabel('P(k)')
    plt.xlabel('k')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)

    degrees = get_graph_degree(g)
    # zamien rozklad na prawdopodobienstwa
    degrees = [x / sum(degrees) for x in degrees]

    # usrednianie:
    if n_graphs > 1:
        degrees = average_graph_degree(degrees, n_graphs, m_0, m, t)

    max_degree = len(degrees)

    # rozklady odniesienia:
    offset = (int(max_degree * (1 - ref_length))) // 2
    x = np.arange(offset, max_degree - offset)
    y = [0 for _ in range(max_degree - 2*offset)]
    y_alpha = [0 for _ in range(max_degree - 2*offset)]
    for x_i in x:
        # rozklad referencyjny:
        y[x_i - offset] = (2 * m * m / (x_i * x_i * x_i))
        # rozklad potegowy:
        y_alpha[x_i - offset] = pow(x_i, -1 * alpha)
           
    plt.plot(x, y, label=r'rozkład referencyjny $(\frac{2*m^2}{k^3})$', linewidth=2) 
    plt.plot(x, y_alpha, label=r'rozkład potegowy o współczynniku $\alpha=%d$' % alpha, linewidth=2) 
    
    # rozklad rzeczywisty:
    x = np.arange(0, max_degree)
    label_deg = r'rozkład $P(k)$ dla $m_0=%d$, $m=%d$' % (m_0, m)
    if n_graphs > 1:
        label_deg = (r'Usredniony po $%d$ ' % n_graphs) + label_deg
    plt.plot(x, degrees, label=label_deg, linewidth=2, marker='o', ls='')
    
    # pokaz:
    plt.legend(loc=3)
    plt.show()
    return

'''
     teoria - propagacja epidemii obliczjac pochodna
'''


def propagate_sis(i_k, beta, gamma, t_max, degrees):

    i_k_vec = []

    # symuluj:
    for _ in np.arange(t_max):

        q_i = calc_q_i(degrees, i_k)
        # print('Q_I=',Q_I)

        dikdt_tab = [0 for _ in i_k]
        for k in range(len(degrees)):
            dikdt = (beta*k*q_i)*(1 - i_k[k]) - gamma*i_k[k]
            dikdt_tab[k] = dikdt

        # print('deriv table t=',t, " : ",dikdt_tab)

        # update pochodnych:
        for k in range(len(degrees)):
            added = i_k[k] + dikdt_tab[k]
            if added > 1:
                i_k[k] = 1
            elif added < 0:
                i_k[k] = 0
            else:
                i_k[k] = added

        # dodaj do wektora
        i_k_vec.append(trim_zeros(i_k, degrees))

    return i_k_vec

'''
    teoria - rozwiazanie rownania rozniczkowego w nieskonczonosci
'''


def infinite_sis(i_k, beta, gamma, degrees):

    q_i = calc_q_i(degrees, i_k)

    i_k_inf = [0 for _ in degrees]
    for k in range(len(degrees)):
        lkq = (beta/gamma) * k * q_i
        i_k_inf[k] = lkq / (1 + lkq)

    return trim_zeros(i_k_inf, degrees)

'''
     symulacja - porpagacja epidemii zmieniajac stany wierzcholkow grafow zgodnie z p-nstwem.
'''


def simulate_sis(graph, beta, gamma, t_max, degrees):
    i_k_vec = []
    for _ in np.arange(t_max):
        for k, v in graph.items():
            # chorzy zdrowieja
            if v[0] == 1 and toss(gamma):
                # print('node',k,'recured')
                v[0] = 0
            # sasiad (sasiedzi) zakaza zdrowych
            if v[0] == 0:
                for adj in v[1]:
                    if graph[adj][0] == 1 and toss(beta):
                        # print('node',k,'got sick.')
                        v[0] = 1
        # dodaj do wektora
        i_k_vec.append(trim_zeros(calc_i_k(graph), degrees))

    return i_k_vec

# '''
#     symulacja - usredniona dla n_sims powtorzen
# '''
#
#
# def simulate_average_sis(graph, beta, gamma, t_max, degrees, n_sims=1):
#     i_k_vec_av = np.array(simulate_sis(graph, beta, gamma, t_max, degrees))
#     if n_sims > 1:
#         for _ in np.arange(n_sims-1):
#             heal_all(graph)
#             # infect(graph, randint(0, len(graph)))
#             infect(graph, 64)
#             i_k_vec = np.array(simulate_sis(graph, beta, gamma, t_max, degrees))
#             i_k_vec_av += i_k_vec
#         i_k_vec_av /= n_sims
#
#     return i_k_vec_av.tolist()

''' 
    wyliczanie Q_I
'''


def calc_q_i(degrees, i_k):
    n = sum(degrees)
    # print('N=',N)
    k_sum = 0
    i = 0
    for d in degrees:
        k_sum += i * d
        i += 1
    # print('ksum=',k_sum)
    k_med = k_sum / n  # <k> = 2*E/n
    # print('k_med=',k_med)
    q_i = 0
    for k in range(len(degrees)):
        # print('Q_k',(k/k_med) * degrees[k] / n)
        q_i += ((k/k_med) * (degrees[k] / n) * i_k[k])
    return q_i


'''
    Braki w rozkladzie reprezentuj jako -1
'''


def trim_zeros(i_k, degrees):
    i_k_trim = []
    # i_k_inf_trim = []
    idx = 0
    for d in degrees:
        if d == 0:
            i_k_trim.append(-1)
        else:
            i_k_trim.append(i_k[idx])
        idx += 1
    return i_k_trim


# def plot_infected(t_vec, infected_number, infected_max):
#
#     plt.figure(3)
#     plt.title(r'Liczba zainfekowanych (I) w czasie, do chwili $t=%d$' % t_vec[-1])
#     plt.ylabel('I')
#     plt.xlabel('t')
#     plt.axis([-t_vec[-1]*0.05, t_vec[-1]*1.05, -infected_max*0.05, infected_max*1.05])
#     plt.grid(True)
#
#     susceptible_number = (infected_max-np.array(infected_number)).tolist()
#     print('infected =', infected_number)
#     print('susceptible =', susceptible_number)
#     print('time vector =', t_vec)
#     plt.plot(t_vec, infected_number, label='zainfekowani', linewidth=2)
#     plt.plot(t_vec, susceptible_number, label='zdrowi', linewidth=2)
#
#     plt.legend(loc=7)
#     plt.show()
#     return


def plot_sis(t, infected_vect_calc, infected_vect_sim, beta, gamma, k):

    plt.figure(k + 10)
    plt.title(r'Prawdopodobienstwo infekcji $i_{%d}(t)$ do chwili $t=%d$ dla $\beta=%f$, $\gamma=%f$'
              % (k, t, beta, gamma))
    plt.ylabel(r'$i_{%d}$' % k)
    plt.xlabel(r'$t$')
    plt.grid(True)
    
    t_vec = range(t)
    calc_vec = [i[k] for i in infected_vect_calc]
    sim_vec = [i[k] for i in infected_vect_sim]

    plt.plot(t_vec, calc_vec, label='rownanie rozniczkowe', linewidth=2)
    plt.plot(t_vec, sim_vec, label='symulacja', linewidth=2, marker='o', ls='')

    plt.legend(loc=7)
    plt.show()
    return


'''
    celowe zakazenie wybranej osoby.
'''


def infect(graph, idx):
    graph[idx][0] = 1
    return

'''
    wyleczenie populacji
'''


def heal_all(graph):
    for idx in np.arange(len(graph)):
        graph[idx+1][0] = 0  # graph[:][0] = 0 nie dziala :(
    return


'''
    liczenie wektora i_k dla calego grafu
'''


def calc_i_k(graph):
    # nodes_count = len(graph)
    degrees = get_graph_degree(graph)
    # wez max. stopien
    max_deg = 0
    for k, v in graph.items():
        if len(v[1]) > max_deg:
            max_deg = len(v[1])
    # print('max_deg=',max_deg)

    # utworz tablice
    i_k_val = np.zeros(len(degrees))
    for k, v in graph.items():
        if v[0] == 1:
            i_k_val[len(v[1])] += 1
    # print(i_k_val)

    # zamien na prawdop.
    i_k_prob = [0 for _ in degrees]
    for d in range(len(degrees)):
        if degrees[d] != 0:
            i_k_prob[d] = i_k_val[d] / degrees[d]
        else:
            i_k_prob[d] = 0
    return i_k_prob


'''
    lambda c z symulacji i teoretyczne
'''


def calc_lambda_c(graph, degrees, i_k, node_idx):
    deg = len(graph[node_idx][1])
    i_k_value = i_k[deg]

    n = sum(degrees)
    k_sum = 0
    i = 0
    for d in degrees:
        k_sum += i * d
        i += 1
    k_med = k_sum / n
    lambda_c_sim = 1 / (k_med * (1 - i_k_value))
    lambda_c_the = k_med / (k_med * k_med)
    return lambda_c_sim, lambda_c_the


def toss(prob):
    val = uniform(0, 1)
    # print('val=',val)
    if val > prob:
        return False
    
    return True


if __name__ == "__main__":
    main()
