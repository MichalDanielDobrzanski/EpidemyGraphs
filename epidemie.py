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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def main():

    # generowanie grafu
    m_0 = 3
    m = 3  # Warunek: m <= m_0
    t = 3400-m_0

    total_nodes = t + m_0

    graph = generate_graph(m_0, m, t)
    # print_graph(graph)
    # plot_graph(graph,graph_layout='spring')
    degree = get_graph_degree(graph, True)
    # plot_graph_degree(graph, m_0, m, t, n_graphs=1)


    # parametry modelu SIS
    beta = 0.03
    gamma = 0.01

    # celowe zarażenie:
    infect(graph,64); # losowy wezel

    # policz i_k dla grafu:
    infected_vect = calc_i_k(graph);
    print(bcolors.WARNING + 'poczatkowy stan sieci: ' + bcolors.ENDC + 'infected_vect=',infected_vect)

    t_max = 100
    print(bcolors.BOLD + 't_max = ' +  str(t_max) + bcolors.ENDC)

    # wylicz rownania rozniczkowe:
    infected_vect, infected_vect_inf, sis_num = propagate_sis(infected_vect, beta, gamma, t_max, degree)
    print(bcolors.HEADER + 'teoria: ' + bcolors.ENDC  + 'infected_vect=',infected_vect)
    print(bcolors.OKGREEN + 'teoria(w niesk.): ' + bcolors.ENDC  + 'infected_vect=',infected_vect_inf)

    # przeprowadz symulacje:
    infected_vect_sim = simulate_sis(graph, beta, gamma, t_max, degree)
    print(bcolors.OKBLUE + 'symulacja: ' + bcolors.ENDC  + 'infected_vect=',infected_vect_sim)

    # print('lengths=',len(infected_vect),len(infected_vect_inf),len(infected_vect_sim))

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
        graph[m_i + 1] = [0, l] # zrob liste
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

    degrees = [0 for x in range(max_degree + 1)]
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

    k_med = sum(degrees) / len(degrees)
    Q_I = 0
    for k in range(len(degrees)):
        Q_I += (k/k_med)*degrees[k]*i_k[k]

    # symuluj:
    for t in np.arange(t_max):
        dikdt_tab = [0 for i in i_k]
        for k in range(len(degrees)):
            dikdt = (beta*k*Q_I)*(1 - i_k[k]) - gamma*i_k[k]
            dikdt_tab[k] = dikdt

        # update pochodnych:
        for k in range(len(degrees)):
            added = i_k[k] + dikdt_tab[k]
            if added > 1:
                i_k[k] = 1
            elif added < 0:
                i_k[k] = 0
            else:
                i_k[k] = added

    # rozwiazanie rownania rozniczkowego w nieskonczonosci:
    i_k_inf = [0 for i in i_k]
    for k in range(len(degrees)):
        lkq = (beta/gamma) * k * Q_I
        i_k_inf[k] = lkq / (1 + lkq)

    i_k_num = (np.round(np.array(i_k_inf)*degrees)).astype(int).tolist()

    # usun miejsca, w ktorych rozklad wierczholkow ma zera
    return trim_zeros(i_k,degrees), trim_zeros(i_k_inf,degrees), i_k_num


'''
    Braki w rozkladzie reprezentuj jako -1
'''
def trim_zeros(i_k,degrees):
    i_k_trim = []
    i_k_inf_trim = []
    idx = 0
    for d in degrees:
        if d == 0:
            i_k_trim.append(-1)
        else:
            i_k_trim.append(i_k[idx])
        idx += 1
    return i_k_trim

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

'''
    celowe zarazenie wybranej osoby.
    Zwraca wektor i_k dla calego grafu.
'''
def infect(graph,idx):
    graph[idx][0] = 1;
    return

'''
    liczenie wektora i_k dla calego grafu
'''
def calc_i_k(graph):
    nodes_count = len(graph)
    degree = get_graph_degree(graph)
    # wez max. stopien
    max_deg = 0;
    for k,v in graph.items():
        if len(v[1]) > max_deg:
            max_deg = len(v[1])
    #print('max_deg=',max_deg)

    # utworz tablice
    I_k = np.zeros(len(degree))
    for k,v in graph.items():
        if v[0] == 1:
            I_k[len(v[1])] += 1
    #print(I_k)

    # zamien na prawdop.
    i_k = np.zeros(len(degree))
    for d in range(len(degree)):
        if degree[d] != 0:
            i_k[d] = I_k[d] / degree[d]
        else:
            i_k[d] = 0
    return i_k


def toss(prob):
    val = random.uniform(0, 1)
    # print('val=',val)
    if val > prob:
        return False
    
    return True

def simulate_sis(graph, beta, gamma, t_max, degrees):
    for t in np.arange(t_max):
        for k,v in graph.items():
            # chorzy zdrowieja
            if v[0] == 1 and toss(gamma):
                # print('node',k,'recured')
                v[0] = 0
            # zdrowi sie zarazaja od sasiada (sasiadow)
            if v[0] == 0:
                for adj in v[1]:
                    if graph[adj][0] == 1 and toss(beta):
                        # print('node',k,'got sick.')
                        v[0] = 1
    return trim_zeros(calc_i_k(graph),degrees)


if __name__ == "__main__":
    main()
