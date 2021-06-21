import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import derivee
import numpy as np

def complete_table(code) : 
    """
    Create a table which contains GDP, Energy and Population data of a country
    code: str, key of "Our World in data" (eg: France: FRA)
    """
    energy_table = pd.read_csv('../BDD/energy.csv', sep=';')
    energy_table = energy_table[energy_table.Code==code]
    gdp_table = pd.read_csv('../BDD/gdp-per-capita-maddison-2020.csv')
    gdp_table = gdp_table[gdp_table.Code==code]
    pop_table = pd.read_csv('../BDD/population.csv', sep=';')
    pop_table = pop_table[pop_table.Code==code]

    complete_table = pd.merge(gdp_table, pop_table, on=['Year'])
    complete_table = pd.merge(complete_table, energy_table, on=['Year'])

    del complete_table['Entity_x']
    del complete_table['Code_x']
    del complete_table['145446-annotations']
    del complete_table['Entity_y']
    del complete_table['Code_y']

    complete_table.rename(columns={'GDP per capita':'GDPC','Total population (Gapminder, HYDE & UN)': 'Population','Primary energy consumption (TWh)': 'Energy'  }, inplace=True)
    complete_table['GDP']= complete_table['GDPC']*complete_table['Population']
    complete_table['EnergyC'] = complete_table['Energy']/complete_table['Population']
    complete_table['Intensity'] = complete_table['Energy']/complete_table['GDP']
    return complete_table


def comparative_plot_bis (countries, x_data=['Year', 'Year', 'Year', 'GDP' ], y_data=['GDP', 'Energy', 'Intensity','Energy' ]) : 
    """
    Print a plot with the data of given countries and with abscissa and ordinates chosen
    countries: table with countries codes, eg: ['FRA', 'CAN', 'USA']
    x_data
    """
    n= len(x_data)
    sns.set_theme(style="whitegrid")
    assert len(x_data)==len(y_data)

    lines = n//2 + n%2
    fig, axs = plt.subplots(lines, 2, figsize= (20, 10*(n//2)), squeeze=False)

    
    for code in countries : 
        table = complete_table(code)
        for i in range(n) :
            
            x_item, y_item = x_data[i], y_data[i]

            assert x_item in table.columns

            if y_item in table.columns : 
                yplot = table[y_item]
                xplot = table[x_item]
                axs[i//2][i%2].plot(xplot, yplot, label = code)
                axs[i//2][i%2].set_ylabel(y_item )
                axs[i//2][i%2].set_xlabel(x_item )
                plt.legend()

            elif y_item == "dE" :
                xplot, yplot = derivee.growth_rate(table[x_item], table['Energy'],s = 10, a = 15 )
                axs[i//2][i%2].plot(xplot, yplot, label = code)
                axs[i//2][i%2].set_title("Energy consumption variations (%)")
                plt.legend()
                

            elif y_item == "dEC" :
                xplot, yplot = derivee.growth_rate(table[x_item], table['EnergyC'],s = 10, a = 15 )
                axs[i//2][i%2].plot(xplot, yplot, label = code)
                axs[i//2][i%2].set_title("Energy per capita consumption variations (%)")
                plt.legend()

            elif y_item == "dGDPC" :
                xplot, yplot = derivee.growth_rate(table[x_item], table['GDPC'],s = 10, a = 15 )
                axs[i//2][i%2].plot(xplot, yplot, label = code)
                axs[i//2][i%2].set_title("GDP per capita variations (%)")
                plt.legend()

            elif y_item == "dGDP" :
                xplot, yplot = derivee.growth_rate(table[x_item], table['GDP'],s = 10, a = 15 )
                axs[i//2][i%2].plot(xplot, yplot, label = code)
                axs[i//2][i%2].set_title("GDP variations (%)")
                plt.legend()
            
        plt.legend()

    
    return None
            


def energy_efficiency(countries = ['FRA', 'GBR', 'CHN', 'DEU', 'CAN', 'JPN',  'USA', 'OWID_WRL' ]) : 
    
    energy_p = pd.read_csv('../BDD/energy.csv', sep=';')
    energy_f = pd.read_csv('../BDD/energy-final.csv', sep=';')

    merged_data= pd.merge(energy_p, energy_f, on=['Year'])
    merged_data = merged_data[merged_data.Code_x == merged_data.Code_y]


    return merged_data


                
                    
                


        
        



