#Parcial 2 Programacion

##Daian Alejandra Bermudez Ceballos 
# Estudiante H

#ControlKyC
#ControlkyU
#Documentacion

import os
import pandas as pd

class CSVReader:
    def _init_(self, file_path=None):
        if file_path is None:
            self.file_path = os.getcwd()  # i no proporciona una ruta, usa la ruta actual
        else:
            self.file_path = file_path
            if not os.path.exists(file_path):
                raise ValueError(f"La ruta proporcionada '{file_path}' no existe")
        self.df = None
    def prompt_file_path(self):
        file_path = input("Por favor, ingrese la ruta del archivo o presione Enter para usar la ruta actual:\n")
        if file_path == "":
            self.file_path = os.getcwd()
        else:
            self.file_path = file_path
            if not os.path.exists(file_path):
                raise ValueError(f"La ruta proporcionada '{file_path}' no existe")

    def choose_file(self):
        files = os.listdir(self.file_path)
        csv_files = [f for f in files if f.endswith('.csv')]
        if not csv_files:
            raise ValueError("No se encontraron archivos CSV en la ruta especificada")
        print("Archivos CSV encontrados:")
        for i, file in enumerate(csv_files):
            print(f"{i+1}. {file}")
        file_indexes = input("Seleccione los archivos que desea abrir separados por comas: ").split(",")
        selected_files = []
        for index in file_indexes:
            file_index = int(index) - 1
            if file_index < 0 or file_index >= len(csv_files):
                raise ValueError("Índice de archivo inválido")
            selected_files.append(os.path.join(self.file_path, csv_files[file_index]))
        return selected_files

    def select_columns(self, data):
        for i, df in enumerate(data):
            print(f"Tabla {i+1}:")
            print(df)
            print("Columnas disponibles:")
            for j, col in enumerate(df.columns):
                print(f"{j+1}. {col}")
            col_indexes = input("Seleccione las columnas que desea mostrar separadas por comas: ").split(",")
            selected_cols = [df.columns[int(index)-1] for index in col_indexes]
            selected_data = df[selected_cols]
            data[i] = selected_data
        merged_data = pd.concat(data, axis=1)
        return merged_data

    def read_csv_files(self):
        csv_files = self.choose_file()
        list_data = []
        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            list_data.append(data)
        merged_data = self.select_columns(list_data)
        return merged_data
    def generate_database(self):
        # Función que genera una nueva base de datos

        # Se crea un dataframe de ejemplo
        data = {'id': [1, 2, 3, 4, 5],
                'nombre': ['Juan', 'Pedro', 'María', 'Luisa', 'Ana'],
                'edad': [25, 30, 27, 23, 28]}
        self.df = pd.DataFrame(data)
        self.df.to_csv('C:/Users\Daian\Desktop\ParciProgra2')

    def print_database(self):
        # Función que imprime la base de datos
        print(self.df)


csv_reader = CSVReader()
csv_reader.prompt_file_path()  # Pregunta al usuario por la ruta del archivo
data = csv_reader.read_csv_files()
print(data)
csv_reader.generate_database()
print(csv_reader.print_database())