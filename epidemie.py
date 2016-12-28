"""

    Program do modelowania epidemii (modelowanych za pomocą modelu SIS)
    na grafach losowych wygenerowanych za pomocą modelu Barabasi-Albert
    
    Instalowanie biblioteki do rysowania grafów:
        sudo pip3 install networkx
    
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


def main():

    # generowanie grafu
    m_0 = 3
    m = 3  # Warunek: m <= m_0
    t = 3500-m_0

    total_nodes = t + m_0

    # parametry modelu SIS
    # i_k0 = 0.001
    i_k_number = 1
    i_k0 = i_k_number/total_nodes
    beta = 0.03
    gamma = 0.05

    graph = generate_graph(m_0, m, t)
    # print_graph(graph)
    degree = get_graph_degree(graph, True)


    infected_vect = [0 for d in degree]  # warunek poczatkowy
    m = 0
    idx = 0
    for i in range(len(degree)):
        if degree[i] > m:
            m = degree[i]
            idx = i
    print('max=', m, 'idx=', idx)
    infected_vect[idx] = i_k0

    # plot_graph(graph,graph_layout='spring')
    # plot_graph_degree(graph, m_0, m, t, n_graphs=1)

    t_vec = [0]
    infected_number = [i_k_number]
    for j in range(50):
        t_0 = j+1
        sis, infected_vect, sis_num = propagate_sis(infected_vect, beta, gamma, t_0, degree)
        # print('prawdop. ze wezel o stopniu k bedzie zakazony w chwili t=', t_0, ': ', sis)
        # print('rozwiazanie rown.rozniczk.=', sis_inf)
        print('liczba zakazonych wezlow o stopniu k w chwili t =', t_0, ': ', sis_num, '; razem:', sum(sis_num),
              '/', total_nodes)
        t_vec.append(t_0)
        infected_number.append(sum(sis_num))

    plot_infected(t_vec, infected_number, total_nodes)

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
        graph[m_i + 1] = (0, l)
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
        p = random.randint(0, total_p)
        cum_p = 0
        for k, v in g.items():
            cum_p += len(v)
            if (not k in marked) and (p <= cum_p):
                marked.append(k)
                # updatuj graf (do wczesniejszego wezla dodaj wskazanie na nowy tworzony):
                g[k][1].append(num_node)
                break
    # updatuj graf (dodaj nowy wezel wraz z jego polaczeniami):
    new_node = {num_node: (0, marked)}
    g.update(new_node)
    return


def get_graph_degree(g, print_deg=False):
    # pobieram rozklad stopni wierzczholkow
    max_degree = 0
    for k, v in g.items():
        degree = len(v[1])
        if degree > max_degree:
            max_degree = degree

    degrees = [0 for x in range(max_degree + 1)]
    for k, v in g.items():
        degree = len(v[1])
        degrees[degree] += 1

    if print_deg:
        print(degrees)

    return degrees


# Utilities:
def print_graph(g):
    print("Graf o", len(g), "wierzcholkach:")
    for k, v in g.items():
        print(k, ':', v)
    return


def plot_graph(g, labels=None, graph_layout='shell',
               node_size=200, node_color='blue', node_alpha=0.3,
               node_text_size=12,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    plt.figure(1)
    plt.title('Graficzne przedstawienie grafu')

    G = nx.Graph()

    for k, v in g.items():
        G.add_node(k)

    for k, v in g.items():
        for i in range(0, len(v[1])):
            G.add_edge(k, v[1][i])

    if graph_layout == 'spring':
        graph_pos = nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos = nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos = nx.random_layout(G)
    else:
        graph_pos = nx.shell_layout(G)
        
    # draw
    nx.draw_networkx_nodes(G, graph_pos, node_size=node_size,
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G, graph_pos, width=edge_tickness,
                           alpha=edge_alpha, edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos, font_size=node_text_size,
                            font_family=text_font)
           
    plt.show()
    return

'''
usrednianie rozkladu stopnia wierzcholka na podstawie kilku grafow
'''


def average_graph_degree(degrees, n_graphs, m_0, m, t):
    if n_graphs > 1:
        for i in range(n_graphs):
            gr = generate_graph(m_0, m, t)
            degs = get_graph_degree(gr)
            # zamien rozklad na prawdopodobienstwa
            degs = [x / sum(degs) for x in degs]

            # wybierz wiekszy rozklad
            l_degs = s_degs = []
            if len(degs) > len(degrees):
                l_degs = degs
                s_degs = degrees
            else:
                l_degs = degrees
                s_degs = degs

            # iteruj po mniejszym
            for i in range(len(s_degs)):
                l_degs[i] += s_degs[i]

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
    y = [0 for i in range(max_degree - 2*offset)]
    y_alpha = [0 for i in range(max_degree - 2*offset)]
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


def propagate_sis(i_k, beta, gamma, t_max, degrees):
    # Q_I = suma (Q(k)*i_k) # r-nie 1
    # Q(k) = k*P(k)/k_med # r-nie 2
    # Q_I = suma (k*P(k)/k_med)*i_k # r-nie 1+2
    # P(k) = get_graph_degree(graph)[k]/k_sum

    # [a, b, c]*[A, B, C] = [aA, bB, cC] -> rozkład * [0, 1, 2, ... max_degree]
    # q_i_k = []

    # s_k = [1 - i for i in i_k]

    # print('t_max=', t_max)
    print('i_k=', i_k)

    # k_sum = sum(degrees)
    # print('ksum=', k_sum)
    k_med = sum(degrees) / len(degrees)
    # print('kmed=', k_med)
    Q_I = 0
    for k in range(len(degrees)):
        Q_I += (k/k_med)*degrees[k]*i_k[k]
    # print('Q_I=', Q_I)

    for t in np.arange(t_max):

        dikdt_tab = [0 for i in i_k]  # wektor pochodnych
        for k in range(len(degrees)):
            dikdt = (beta*k*Q_I)*(1 - i_k[k]) - gamma*i_k[k]  # r-nie 4
            dikdt_tab[k] = dikdt
        # print('wektor pochodnych dla t=',t,':',dikdt_tab)

        # update pochodnych:
        for k in range(len(degrees)):
            added = i_k[k] + dikdt_tab[k]
            if added > 1:
                i_k[k] = 1
            elif added < 0:
                i_k[k] = 0
            else:
                i_k[k] = added

    # rozwiazanie rownania rozniczkowego:
    i_k_inf = [0 for i in i_k]
    for k in range(len(degrees)):
        lkq = (beta/gamma) * k * Q_I
        i_k_inf[k] = lkq / (1 + lkq)

    i_k_num = (np.round(np.array(i_k_inf)*degrees)).astype(int).tolist()

    return i_k, i_k_inf, i_k_num


def plot_infected(t_vec, infected_number, infected_max):

    plt.figure(3)
    plt.title(r'Liczba zainfekowanych (I) w czasie, do chwili $t=%d$' % t_vec[-1])
    plt.ylabel('I')
    plt.xlabel('t')
    plt.axis([-t_vec[-1]*0.05, t_vec[-1]*1.05, -infected_max*0.05, infected_max*1.05])
    plt.grid(True)

    susceptible_number = (infected_max-np.array(infected_number)).tolist()
    print('infected =', infected_number)
    print('susceptible =', susceptible_number)
    print('time vector =', t_vec)
    plt.plot(t_vec, infected_number, label='zainfekowani', linewidth=2)
    plt.plot(t_vec, susceptible_number, label='zdrowi', linewidth=2)

    plt.legend(loc=7)
    plt.show()
    return

if __name__ == "__main__":
    main()
