import sqlite3

miConexion = sqlite3.connect("Favoritos")
miCursor = miConexion.cursor()

# create Favoritos table
miCursor.execute("""
    CREATE TABLE Favoritos (
        id INTEGER PRIMARY KEY,
        Nombre VARCHAR(10),
        Presupuesto DECIMAL,
        Cumplimiento DECIMAL
        )
""")

# insert data into VENTAS table
datosVentas = [
    (1, "Muebles", "15.45", "19.60"),
    (2, "Organizadores", "6.38", "4.47"),
    (3, "Menaje", "9.68", "5.05"),
    (4, "Decoración", "12.91", "9.45"),
    (5, "Aire Libre", "8.84", "5.23")
]
miCursor.executemany("INSERT INTO VENTAS VALUES (?,?,?,?)", datosVentas)

# create PORCENTAJE table
miCursor.execute("""
    CREATE TABLE PORCENTAJE (
        id INTEGER PRIMARY KEY,
        Porcentaje DECIMAL
        )
""")

# insert data into PORCENTAJE table
datosPorcentaje = [
    (1, "126.00"),
    (2, "70.00"),
    (3, "52.00"),
    (4, "73.00"),
    (5, "59.00")
]
miCursor.executemany("INSERT INTO PORCENTAJE VALUES (?,?)", datosPorcentaje)

# join tables using SQL query
miCursor.execute("""
    SELECT VENTAS.id, VENTAS.Familia, VENTAS.Presupuesto, VENTAS.Cumplimiento, PORCENTAJE.Porcentaje
    FROM VENTAS JOIN PORCENTAJE ON VENTAS.id = PORCENTAJE.id
""")

# get results and print them
resultados = miCursor.fetchall()

# print headers
print("ID".ljust(4), "Familia".ljust(15), "Presupuesto".ljust(15), "Cumplimiento".ljust(15), "Porcentaje".ljust(15))

for fila in resultados:
    # print row data
    print(str(fila[0]).ljust(4), fila[1].ljust(15), str(fila[2]).ljust(15), str(fila[3]).ljust(15), str(fila[4]).ljust(15))

# close connection
miConexion.close()

