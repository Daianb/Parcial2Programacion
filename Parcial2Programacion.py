#Parcial 2 Programacion

##Daian Alejandra Bermudez Ceballos 
# Estudiante H

#ControlKyC
#ControlkyU
#Documentacion

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
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
                raise ValueError(f"La ruta proporcionada '{file_path}' no existe")

    def prompt_file_path(self):
        """ Metodo  que pide al usuario la ruta del archivo o con Enter usa la ruta actual """
        file_path = input("\nPor favor, ingrese la ruta del archivo o presione Enter para usar la ruta actual:\n")
        if file_path == "":
            self.file_path = os.getcwd()
        else:
            self.file_path = file_path
            if not os.path.exists(file_path):
                raise ValueError(f"La ruta proporcionada '{file_path}' no existe")

    def choose_file(self):
        """ Método que lista los archivos CSV en la ruta y permite al usuario seleccionar uno o varios """
        files = os.listdir(self.file_path)
        csv_files = [f for f in files if f.endswith('.csv')]
        if not csv_files:
            raise ValueError("No se encontraron archivos CSV en la ruta especificada")
        print("Archivos CSV encontrados: ")
        for i, file in enumerate(csv_files):
            print(f"{i+1}. {file}")
        file_indexes = input("\nSeleccione los archivos que desea abrir separados por espacio: ").split(" ")
        selected_files = []
        for index in file_indexes:
            file_index = int(index) - 1
            if file_index < 0 or file_index >= len(csv_files):
                raise ValueError("Índice de archivo inválido")
            selected_files.append(os.path.join(self.file_path, csv_files[file_index]))
        return selected_files

    def get_selected_columns(self, data):
        """ Metodo que muestra las columas disponibles en un DataFrame al usuario y permite seleccionar algunas """
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
        """ Metodo que seleciona las columas deseeadas por el usuario las combina en un único DataFrame """
        selected_cols = self.get_selected_columns(data)
        selected_data = []
        for i, cols in enumerate(selected_cols):
            selected_data.append(data[i][cols])
        merged_data = pd.concat(selected_data, axis=1)
        return merged_data

    def read_csv_files(self):
        """ Método que unifica y lee los archivos CSV elegidos por el usuario """
        csv_files = self.choose_file()
        list_data = []
        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            list_data.append(data)
        merged_data = self.select_columns(list_data)
        merged_data = self.homogenize_values(merged_data)
        return merged_data

    def fit_linear_regression(self, data):
        """ Metodo para realizar la regresión lineal de las columnas numéricas en el archivo CSV y 
        retornar los coeficientes de regresión lineal"""
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
        """Normaliza los valores en las columnas que contienen cadenas de texto"""
        for col in data.columns:
            if data[col].dtype == 'O':  # Si la columna contiene valores tipo str
                unique_values = data[col].str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').unique()
                # Se convierten los valores a minúsculas, se normalizan para eliminar las tildes y se eliminan los caracteres no ASCII
                print(f"Valores únicos en la columna '{col}':")
                for i, value in enumerate(unique_values):
                    print(f"{i+1}. {value}")
                join_values = input("\n¿Desea unir algunos de estos valores? Indique los índices separados por comas (o presione Enter si no desea unir): ")
                if join_values != "":
                    join_values = [int(index)-1 for index in join_values.split(",")]
                    new_value = input("\nIngrese el nuevo valor que reemplazará a los valores seleccionados: ")
                    for index in join_values:
                        data[col] = data[col].replace(unique_values[index], new_value)
        return data
    
csv_reader = CSVReader()
csv_reader.prompt_file_path()
data = csv_reader.read_csv_files()
print("\nDatos combinados:")
print(data)

coefficients = csv_reader.fit_linear_regression(data)
print("\nCoeficientes de regresión lineal:")
for coeff in coefficients:
    print(f"{coeff[0]} vs. {coeff[1]}: {coeff[2]}")

# Grafica los valores de la tabla
plt.scatter(data.iloc[:,0], data.iloc[:,1])
for coeff in coefficients:
    x = data[coeff[0]]
    y = data[coeff[1]]
    reg_line = coeff[2]*x + coeff[3] # línea de regresión
    #plt.plot(x, reg_line, label=f"Regresión {coeff[0]} vs {coeff[1]}")
    plt.plot(x, reg_line, color='red', label=f"Regresión {coeff[0]} vs {coeff[1]}")

# Agrega título y leyendas al gráfico
plt.title('\nValores de la tabla y regresión lineal')
plt.xlabel('\nVariable independiente')
plt.ylabel('\nVariable dependiente')
plt.legend()

# Muestra el gráfico
plt.show()