# Parcial 2 Programacion

# Daian Alejandra Bermudez Ceballos
# Estudiante H

# ControlKyC
# ControlkyU
# Documentacion

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
        """ Constructor de la clase. Inizializa la ruta del archivo """
        if file_path is None:
            self.file_path = os.getcwd()  # Si no se proporciona una ruta, usa la ruta actual
        else:
            self.file_path = file_path
            if not os.path.exists(file_path):
                raise ValueError(
                    f"La ruta proporcionada '{file_path}' no existe")

    def prompt_file_path(self):
        """ Metodo  que pide al usuario la ruta del archivo o con Enter usa la ruta actual """
        file_path = input(
            "\nPor favor, ingrese la ruta del archivo o presione Enter para usar la ruta actual:\n")
        if file_path == "":
            self.file_path = os.getcwd()
        else:
            self.file_path = file_path
            if not os.path.exists(file_path):
                raise ValueError(
                    f"La ruta proporcionada '{file_path}' no existe")

    def get_file_csv(self):
        """ Método que lista los archivos CSV en la ruta """
        files = os.listdir(self.file_path)
        csv_files = [f for f in files if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(
                "No se encontraron archivos CSV en la ruta especificada")
        print("Archivos CSV encontrados: ")
        for i, file in enumerate(csv_files):
            print(f"{i+1}. {file}")
        return csv_files


    def choose_file(self):
        """ Metodo permite al usuario seleccionar uno o varios archivos tipo csv """
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

    #Decorador 1
    def log_decorator(func):
        """  esta función decoradora se puede utilizar para agregar declaraciones de registro adicionales 
        a cualquier función en Python simplemente decorando esa función con la función log_decorator."""
        def wrapper(*args, **kwargs):
            print("***********************************")
            print(f"Ejecutando función: {func.__name__}")
            print("***********************************")
            result = func(*args, **kwargs)
            print(f"Resultado: {result}")
            return result
        return wrapper  

    @log_decorator
    def join_data(self):
        """  tiene como objetivo combinar dos archivos CSV en un solo marco de datos.
        El método comienza llamando al método get_"""
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

    @log_decorator
    def get_selected_columns(self, data):
        """ Metodo que muestra las columas disponibles en un DataFrame al usuario y permite seleccionar algunas"""
        selected_cols = []
        for i, df in enumerate(data):
            print(f"\nTabla {i+1}:")
            print(df)
            print("\nColumnas disponibles:")
            for j, col in enumerate(df.columns):
                print(f"{j+1}. {col}")
            col_indexes = input(
                "\nSeleccione las columnas que desea mostrar separadas por espacio: ").split(" ")
            selected_cols.append([df.columns[int(index)-1]
                                 for index in col_indexes])
        return selected_cols
    
    @log_decorator
    def select_columns(self, data):
        """ Metodo que seleciona las columas deseeadas por el usuario las combina en un único DataFrame """
        selected_cols = self.get_selected_columns(data)
        selected_data = []
        for i, cols in enumerate(selected_cols):
            selected_data.append(data[i][cols])
        merged_data = pd.concat(selected_data, axis=1)
        return merged_data

    @log_decorator
    def read_csv_files(self):
        """ Metodo que unifica y lee los archivos CSV elegidos y devuelve una lista de columnas seleccionadas por el usuario."""
        csv_files = self.choose_file()
        list_data = []
        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            list_data.append(data)
        merged_data = self.select_columns(list_data)
        merged_data = self.homogenize_values(merged_data)
        return merged_data
    

#Punto 5
    @staticmethod
    def homogenize_values(data):
        """Normaliza los valores en las columnas que contienen cadenas de texto
        Este método recorre todas las columnas del conjunto de datos que se pasa como argumento y para aquellas que contienen valores de tipo cadena (str), 
        permite al usuario fusionar algunos de los valores únicos en la columna en uno solo."""
        for col in data.columns:
            if data[col].dtype == 'O':  # Si la columna contiene valores tipo str
                unique_values = data[col].unique() # Se convierten los valores a minúsculas, se normalizan para eliminar las tildes y se eliminan los caracteres no ASCII
                print(f"Valores únicos en la columna '{col}':")
                for i, value in enumerate(unique_values):
                    print(f"{i+1}. {value}")
                join_values = input(
                    "\n¿Desea unir algunos de estos valores? Indique los índices separados por espacio (o presione Enter si no desea unir): ")
                if join_values != "":
                    try:
                        join_values = [
                            int(index)-1 for index in join_values.split(" ")]
                        print(join_values)
                        new_value = input(
                            "\nIngrese el nuevo valor que reemplazará a los valores seleccionados: ")
                        for index in join_values:
                            data[col] = data[col].replace(
                                unique_values[index], new_value)
                    except:
                        print("Errorsiiiitoooo")
        return data
    
#Punto 14
    @log_decorator
    def solveEq(self, eq, n):
        """  Metodo que resuelve la ecuación utilizando la biblioteca SymPy y traza la función resultante utilizando Matplotlib 
        Resuelve ecuaciones y muestra una gráfica de la solución.
        El método toma dos argumentos: eq, que es la ecuación a resolver como una cadena de texto, y n,
        que es el rango de valores en el eje X que se quiere mostrar en la gráfica."""
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
            print("Errorsiiiitoooo ")

    @log_decorator
    def getData(self, data):
        """ Metodo que toma un dataframe de pandas como entrada y devuelve una lista que contiene el dataframe, las columnas seleccionadas y la columna objetivo seleccionada. 
        Si no se selecciona una columna objetivo, se devuelve una lista vacía."""
        selected_cols = self.get_selected_columns([data])
        selected_target = self.get_selected_columns([data])
        if len(selected_target) < 2:
            return [data, selected_cols, selected_target]
        else:
            self.getData(data)
            
#Punto8
    def filterDataExact(self, data):
        """ El método toma un dataframe de pandas como entrada y filtra los datos según los valores exactos seleccionados por el usuario en una columna. 
        Si la columna no es de tipo objecto o si se seleccionan varias columnas, se llama recursivamente al método getData."""
        col = self.get_selected_columns([data])
        # selected_target = self.get_selected_columns([data])
        # print(data[col[0]].dtype == 'O')
        print(col)
        if len(col) < 2 and data[col[0][0]].dtype == 'O':
            vals = input(
                "\nSeleccione los valores que desea filtrar separadas por espacio, exacto: ").split(" ")
            print(vals)
            return data.apply(lambda row: row[data[col[0][0]].isin(vals)])
            # return col , vals
        else:
            self.getData(data)

    
    def filterDataContain(self, data):
        """ El método toma un dataframe de pandas como entrada y filtra los datos según los valores que contengan una cadena de caracteres específica seleccionada por el usuario en una columna. 
        Si la columna no es de tipo objecto o si se seleccionan varias columnas, se llama recursivamente al método getData."""
        col = self.get_selected_columns([data])
        # selected_target = self.get_selected_columns([data])
        if len(col) < 2 and data[col[0][0]].dtype == 'O':
            vals = input(
                "\nSeleccione los valores que desea filtrar separadas por espacio, contengan: ").split(" ")
            print(vals)
            return data.apply(lambda row: row[data[col[0][0]].str.contains(vals[0])])
            # return col , vals
        else:
            self.getData(data)

    
    def filterDataStart(self, data):
        """ El método toma un dataframe de pandas como entrada y filtra los datos según los valores que empiecen por una cadena de caracteres específica seleccionada por el usuario en una columna. 
        Si la columna no es de tipo objecto o si se seleccionan varias columnas, se llama recursivamente al método getData."""
        col = self.get_selected_columns([data])
        # selected_target = self.get_selected_columns([data])
        if len(col) < 2 and data[col[0][0]].dtype == 'O':
            vals = input(
                "\nSeleccione los valores que desea filtrar separadas por espacio, inicien por: ").split(" ")
            print(vals)
            if len(vals) < 2:
                return data.apply(lambda row: row[data[col[0][0]].str.startswith(vals[0])])
            # return col , vals
            else:
                self.getData(data)
        else:
            self.getData(data)
        # Graficas
    # df[df['Courses'].str.contains("Spark")]

#Fin Punto8

#Punto12
    def regreLinealSimple(self, data, xc, yt):
        """ Metodo para realizar la regresión lineal  simple entre dos variables de las columnas numéricas en el archivo CSV y 
        retornar los coeficientes de regresión lineal
        se utiliza la clase LinearRegression de scikit-learn para realizar la regresión lineal. """
        fig, ax = plt.subplots(figsize=(6, 3.84))

        data.plot(
            x=xc,
            y=yt,
            c='firebrick',
            kind="scatter",
            ax=ax
        )
        ax.set_title('Distribución de bateos y runs')
        x = data[[xc]].values.reshape(-1, 1)
        y = data[[yt]]
        reg = LinearRegression().fit(x, y)
        # print("Coeficiente de determinación R^2:", reg.score(x, y))
        reg_line = reg.coef_[0]*x + reg.intercept_[0]  # línea de regresión
        # plt.plot(x, reg_line, label=f"Regresión {coeff[0]} vs {coeff[1]}")
        plt.plot(x, reg_line, color='red',
                 label=f"Regresión {round(reg.coef_[0][0],2)}*x + {round(reg.intercept_[0],2)}, R^2={round(reg.score(x, y),2)}")
        # Agregar título y leyendas al gráfico
        plt.title('\nValores de la tabla y regresión lineal')

        plt.legend()
        plt.show()

#Decorador
def plot_graph(func):
    """ El decorador plot_graph es utilizado para trazar una línea de regresión lineal simple
    en las columnas seleccionadas utilizando el método regreLinealSimple de otra clase CsvReader."""
    def wrapper(*args, **kwargs):
        # Call the original function
        result = func(*args, **kwargs)
        print(result[1])
        for i in result[1]:
            csv_reader.regreLinealSimple(result[0], i, result[2])

        return result
    return wrapper


if __name__ == '__main__':
    
    csv_reader = CSVReader()
    csv_reader.prompt_file_path()
    data = csv_reader.read_csv_files()
    data = data.dropna()
    @plot_graph
    def generate_data(data):
        info = csv_reader.getData(data)
        return [info[0], info[1][0], info[2][0][0]]

    generate_data(data)
    filterA = csv_reader.filterDataExact(data)
    print(filterA)
    filterB = csv_reader.filterDataContain(data)
    print(filterB)
    filterC = csv_reader.filterDataStart(data)
    print(filterC)
    ecu = input("Ecu: ")
    def eq(x): return x
    csv_reader.solveEq(eq(ecu), 3)



#En general, estos métodos proporcionan una forma fácil y flexible de seleccionar y procesar datos en un dataframe de pandas. 
#La biblioteca SymPy y Matplotlib se utilizan para realizar operaciones matemáticas y visualizar los resultados de manera efectiva.
