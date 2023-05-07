#Parcial 2 Programacion

##Daian Alejandra Bermudez Ceballos 
# Estudiante H

#ControlKyC
#ControlkyU
#Documentacion

import numpy as np
import matplotlib.pyplot as plt
from sympy import sympify
from sympy.plotting import plot
from sympy import symbols
from sympy import lambdify
from sympy.utilities.lambdify import implemented_function
from sympy.abc import x
from sympy import Symbol
from sympy.solvers import solve
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt 

class CSVReader:
    """ Clase que lee archivos csv """

    def init(self, file_path=None):
        """ Metodo inicial para proporcionar o encontrar la ruta del documento"""
        if file_path is None:
            self.file_path = os.getcwd()  # Si no se proporciona una ruta, usa la ruta actual
        else:
            self.file_path = file_path
            if not os.path.exists(file_path):
                raise ValueError(f"La ruta proporcionada '{file_path}' no existe")

    def prompt_file_path(self):
        """ Metodo que lee la ruta """
        file_path = input("\nPor favor, ingrese la ruta del archivo o presione Enter para usar la ruta actual:\n")
        if file_path == "":
            self.file_path = os.getcwd()
        else:
            self.file_path = file_path
            if not os.path.exists(file_path):
                raise ValueError(f"La ruta proporcionada '{file_path}' no existe")

    def get_file_csv(self):
        """ Metodo que escoje el archivo tipo csv """
        files = os.listdir(self.file_path)
        csv_files = [f for f in files if f.endswith('.csv')]
        if not csv_files:
            raise ValueError("No se encontraron archivos CSV en la ruta especificada")
        print("Archivos CSV encontrados: ")
        for i, file in enumerate(csv_files):
            print(f"{i+1}. {file}")
        return csv_files
    def choose_file(self):
        """ Metodo que escoje el archivo tipo csv """
        csv_files = self.get_file_csv()
        file_indexes = input(
            "\nSeleccione los archivos que desea abrir separados por espacio: ").split(" ")
        selected_files = []
        for index in file_indexes:
            file_index = int(index) - 1
            if file_index < 0 or file_index >= len(csv_files):
                raise ValueError("Índice de archivo inválido")
            selected_files.append(os.path.join(
                self.file_path, csv_files[file_index]))
        return selected_files
    
    def join_data(self):
        datL = self.get_file_csv()
        BD = []
        for i in datL:
            k = pd.read_csv(i)
            # k.head()
            k = pd.DataFrame(k)
            BD.append(k)
        a = set(BD[0].columns)
        b = set(BD[1].columns)
        on_dat = a & b
        print(on_dat)
        print(a.intersection(b))
        print(list(on_dat)[0])
        return pd.merge(BD[0], BD[1], how='inner', on=list(on_dat)[0])
        

    def get_selected_columns(self, data):
        """ Metodo que muestra las columas disponibles al usuario"""
        selected_cols = []
        for i, df in enumerate(data):
            print(f"\nTabla {i+1}:")
            print(df)
            print("\nColumnas disponibles:")
            for j, col in enumerate(df.columns):
                print(f"{j+1}. {col}")
            col_indexes = input("\nSeleccione las columnas que desea mostrar separadas por espacio: ").split(" ")
            selected_cols.append([df.columns[int(index)-1] for index in col_indexes])
        return selected_cols

    def select_columns(self, data):
        """ Metodo que seleciona las columas deseeadas por el usuario"""
        selected_cols = self.get_selected_columns(data)
        selected_data = []
        for i, cols in enumerate(selected_cols):
            selected_data.append(data[i][cols])
        merged_data = pd.concat(selected_data, axis=1)
        return merged_data

    def read_csv_files(self):
        """ Metodo que unifica y lee los archivos"""
        csv_files = self.choose_file()
        list_data = []
        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            list_data.append(data)
        merged_data = self.select_columns(list_data)
        merged_data = self.homogenize_values(merged_data)
        return merged_data

    def fit_linear_regression(self, data):
        """ Metodo para creae la regresion lineal"""
        selected_cols = self.get_selected_columns([data])
        coefficients = []
        for cols in selected_cols:
            if not all(data[col].apply(lambda x: str(x).isdigit()).all() for col in cols):
                print(f"\nNo se puede hacer regresión lineal a la columna {cols} ya que contiene datos no numéricos")
                continue
            X = data[cols[0]].values.reshape(-1, 1)
            y = data[cols[1]].values.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            coefficients.append((cols[0], cols[1], reg.coef_[0][0], reg.intercept_))
        return coefficients
    
    def homogenize_values(self, data):
        
        for col in data.columns:
            if data[col].dtype == 'O':  # Si la columna contiene valores tipo str
                unique_values = data[col].unique()
                # Se convierten los valores a minúsculas, se normalizan para eliminar las tildes y se eliminan los caracteres no ASCII
                print(f"Valores únicos en la columna '{col}':")
                for i, value in enumerate(unique_values):
                    print(f"{i+1}. {value}")
                join_values = input("\n¿Desea unir algunos de estos valores? Indique los índices separados por comas (o presione Enter si no desea unir): ")
                if join_values != "":
                    try:
                        join_values = [int(index)-1 for index in join_values.split(",")]
                        print(join_values)
                        new_value = input("\nIngrese el nuevo valor que reemplazará a los valores seleccionados: ")
                        for index in join_values:
                            data[col] = data[col].replace(unique_values[index], new_value)
                    except:
                        print("Error")
        return data
    
    #  Graficas
    def get_Violin(self, join):

        sns.boxplot(x=join[join.columns[0]],
                    y=join[join.columns[3]], hue=join[join.columns[4]])
        plt.xticks(rotation=90)
        # # Mostrar el gráfico
        plt.show()

    def solveEq(self, eq, n):
        
        try:
            expr = sympify(eq)
            sol = solve(expr, x)
            evasol = [expr.subs(x, i) for i in sol]
            print(expr)
            print(solve(expr, x))
            print(evasol)

            xx = np.linspace(-n, n, 1000)
            yy = lambdify(x, [expr])(xx)
            plt.title(expr)
            plt.plot(xx, np.transpose(yy))
            plt.plot(sol, evasol, 'k*')
            plt.axvline(x=0, color='k')
            plt.axhline(y=0, color='k')
            plt.show()
        except:
            print("Error")
           

# ecu = input("Ecu: ")
# eq = lambda x: x

csv_reader = CSVReader()
# csv_reader.solveEq(eq(ecu), 3)  
csv_reader.prompt_file_path()



# join = csv_reader.join_data()
# print(join)
# print(join.describe())
# csv_reader.read_csv_files()
data = csv_reader.read_csv_files()
print(data)

print("Datos combinados:")
print(data)
print(data.describe())
coefficients = csv_reader.fit_linear_regression(data)
print("\nCoeficientes de regresión lineal:")
for coeff in coefficients:
    print(f"{coeff[0]} vs. {coeff[1]}: {coeff[2]}")
import matplotlib.pyplot as plt
# Graficar los valores de la tabla
plt.scatter(data.iloc[:,0], data.iloc[:,1])
for coeff in coefficients:
    x = data[coeff[0]]
    y = data[coeff[1]]
    reg_line = coeff[2]*x + coeff[3] # línea de regresión
    #plt.plot(x, reg_line, label=f"Regresión {coeff[0]} vs {coeff[1]}")
    plt.plot(x, reg_line, color='red', label=f"Regresión {coeff[0]} vs {coeff[1]}")
# Agregar título y leyendas al gráfico
plt.title('\nValores de la tabla y regresión lineal')
plt.xlabel('\nVariable independiente')
plt.ylabel('\nVariable dependiente')
plt.legend()
plt.show()
