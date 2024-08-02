import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Função para simular um segundo da população
def simulate_population_step(population, birth_rate, death_rate):
    births = np.random.poisson(birth_rate)
    deaths = np.random.poisson(death_rate)
    net_change = births - deaths
    new_population = population + net_change
    
    return new_population, births, deaths

# Função para calcular as estatísticas
def compute_statistics(data):
    mean = np.mean(data)
    #mode = stats.mode(data)[0][0]
    std_dev = np.std(data)
    variance = np.var(data)
    
    return mean, std_dev, variance

# Função para realizar a regressão linear
def perform_regression(time, population):
    X = sm.add_constant(time)
    model = sm.OLS(population, X).fit()
    return model

# Configuração da interface do Streamlit
st.title("Simulação de População com Atualizações ao Vivo")

initial_population = st.number_input("População Inicial", value=1000, min_value=1)
birth_rate = st.slider("Taxa de Nascimento (por segundo)", 0.0, 5.0, 1.0)
death_rate = st.slider("Taxa de Mortalidade (por segundo)", 0.0, 5.0, 0.5)
seconds = st.number_input("Duração da Simulação (segundos)", value=100, min_value=1)

if st.button("Iniciar Simulação"):
    time_data = []
    population_data = []
    births_data = []
    deaths_data = []

    population = initial_population
    
    for second in range(seconds):
        population, births, deaths = simulate_population_step(population, birth_rate, death_rate)
        
        time_data.append(second)
        population_data.append(population)
        births_data.append(births)
        deaths_data.append(deaths)

        mean,  std_dev, variance = compute_statistics(population_data)
        
        st.write(f"Tempo: {second + 1}s")
        st.write(f"População Atual: {population}")
        st.write(f"Nascimentos no último segundo: {births}")
        st.write(f"Mortes no último segundo: {deaths}")
        st.write(f"Média da População: {mean}")
        st.write(f"Moda da População: {mode}")
        st.write(f"Desvio Padrão da População: {std_dev}")
        st.write(f"Variância da População: {variance}")

        # Atualizar gráficos
        df = pd.DataFrame({
            "Tempo": time_data,
            "População": population_data,
            "Nascimentos": births_data,
            "Mortes": deaths_data
        })

        fig, ax = plt.subplots()
        sns.lineplot(x='Tempo', y='População', data=df, ax=ax, label='População')
        sns.lineplot(x='Tempo', y='Nascimentos', data=df, ax=ax, label='Nascimentos')
        sns.lineplot(x='Tempo', y='Mortes', data=df, ax=ax, label='Mortes')

        ax.set_title('Simulação de População ao Vivo')
        ax.legend()

        st.pyplot(fig)
        
        time.sleep(1)  # Esperar um segundo antes de atualizar novamente

    model = perform_regression(time_data, population_data)
    st.write(model.summary())
    
    fig, ax = plt.subplots()
    sns.regplot(x='Tempo', y='População', data=df, ax=ax, label='População', line_kws={"color":"r","alpha":0.7,"lw":2})

    ax.set_title('Regressão Linear da População')
    ax.legend()

    st.pyplot(fig)
