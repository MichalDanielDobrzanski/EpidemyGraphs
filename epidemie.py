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
    # my code here
    m_0 = 3
    m = 3 # Warunek: m <= m_0
    t = 3500
    
    graph = generate_graph(m_0, m, t)
    #print_graph(graph)
    get_graph_degree(graph)
    #plot_graph(graph,graph_layout='spring')
    plot_graph_degree(graph, m_0, m, t)
    
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

def get_graph_degree(g,print_deg=False):
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
def average_graph_degree(degrees,n_graphs,m_0, m, t):
    if n_graphs > 1:
        for i in range(n_graphs):
            gr = generate_graph(m_0, m, t)
            degs = get_graph_degree(gr)
            degs = [x / len(gr) for x in degs]

            # wybierz wiekszy rozklad
            l_degs = s_degs = []
            if len(degs) > len(degrees):
                l_degs = degs
                s_degs = degrees
            else:
                l_degs = degrees
                s_degs = degs

            # iteruj po wiekszym
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
def plot_graph_degree(g, m_0, m, t, n_graphs=5, alpha=3,ref_length=0.8):
    
    plt.figure(2)
    plt.title(r'Rozkład stopni wierzchołków P(k) w sieci BA dla chwili $t=%d$' % t)
    plt.ylabel('P(k)')
    plt.xlabel('k')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)

    degrees = get_graph_degree(g)
    # zamien rozklad na prawdopodobienstwa
    degrees = [x / len(g) for x in degrees]

    # usrednianie:
    if n_graphs > 1:
        degrees = average_graph_degree(degrees,n_graphs,m_0, m, t)

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
        y_alpha[x_i - offset] = pow(x_i,-1 * alpha)
           
    plt.plot(x, y, label=r'rozkład referencyjny $(\frac{2*m^2}{k^3})$', linewidth=2) 
    plt.plot(x, y_alpha, label=r'rozkład potegowy o współczynniku $\alpha=%d$' % alpha, linewidth=2) 
    
    # rozklad rzeczywisty:
    x = np.arange(0, max_degree)
    label_deg = r'rozkład $P(k)$ dla $m_0=%d$, $m=%d$' % (m_0,m)
    if n_graphs > 1:
    	label_deg = (r'Usredniony po $%d$ ' % n_graphs) + label_deg
    plt.plot(x, degrees, label=label_deg, linewidth=2, marker='o', ls='')
    
    # pokaz:
    plt.legend(loc=3)
    plt.show()
    return

if __name__ == "__main__":
    main()
