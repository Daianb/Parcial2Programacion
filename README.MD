# Parcial2Programacion


# Daian Alejandra Bermudez Ceballos
## Estudiante H
### Ejercicios 1 - 4 - 5 - 8 - 11 - 12 - 14 - 16 - 17 - 18 - 19

#### 1. Importar datos
1.Construya una clase que pueda recibir como argumentos al momento de definirla
el nombre de uno o varios archivos y opcionalmente la ubicación de cada uno. Si no se
ingresa la ruta de algun archivo el programa supone que es la ruta actual y lee todos los archivos de datos. Si no se le ingresa el nombre de ningún archivo el programa debe
generar la base de datos.

4.Como en el ejercicio 3, pero en lugar de unir todas las columnas en una sola tabla, el
programa le muestra al usuario las columnas de las dos tablas y el usuario selecciona
las columnas que necesita. La tabla que se guarda debe tener solo esas columnas. Si lo
resuleve usando SQL no debe hacer los ejercicios de procesar datos

#### 2. Procesar datos
5.La clase debe tener un método que detecte automátiamente en las columnas con valores
tipo str los datos que se identifican como diferentes de forma errada por cuestión de
tildes o mayúsculas/minúsculas y homogeneizar los valores. Además le debe mostrar al
usuario los valores únicos y permitirle decir si desea unir o no algunos de esos valores.

8.La clase debe tener un método que reciba condiciones para los datos tipo cadena y filtre
los datos de la tabla con base en esa condición

11.La clase debe tener un método que calcule los coeficientes de regresión lineal para dos
columnas elegidas por el usuario. (Si hace regresión multi-lineal gana puntos adicionales,
si la hace multi-lineal con regularización, gana aún más puntos adicionales)

#### 3. Graficar
12.Grafique el resultado del numeral anterior. En scatter los valores de la tabla y en línea
la regresión lineal.


14.Realice un método o una función que al recibir una función matemática definida a
partir de una lambda, encuentre los cortes con el eje x, grafique la función y ponga en
scatter los puntos solución.

#### 4. Requisitos generales
16.Defina y use al menos dos decoradores

17.Use *args y **kwargs

18.Documente TODAS las funciones y métodos

Si se pide que construya una clase que pueda hacer algo, debe implementar un ejemplo de
uso en el código.
El código puede ser en formato .py o .ipynb


# Desarrollo

### Antes de empezar debo aclarar que los puntos no estan en orden, ya que se decidió que se irian haciendo por conveniencia y nivel de facilidad para la programadora, gracias por entender, besitos y abracitos.

Comencé exportando las librerias que necesitaba para empezar el **Punto 1**, cree el constructor de la clase que inicializa la ruta del archivo, seguido del Metodo  que pide al usuario la ruta del archivo o con Enter usa la ruta actual, para pasar a la clase **get_file_csv** que lista los archivos CSV en la ruta para continuar con el metodo que permite al usuario seleccionar uno o varios archivos tipo csv 

**Punto 16** con mi primer *decorador* que consiste en que log_decorator toma como argumento la función que se va a decorar y devuelve una nueva función llamada wrapper. Esta función wrapper se utiliza para envolver la función original y agregar algunas declaraciones de registro adicionales antes y después de la ejecución de la función original.

**El Punto 17** con la función wrapper toma argumentos arbitrarios *args y **kwargs, lo que significa que puede envolver cualquier función sin saber de antemano cuántos argumentos tomará la función original. Primero, imprime un conjunto de líneas de asteriscos para separar la salida en el registro. Luego, imprime el nombre de la función original utilizando la variable func.__ name __ y continua con otra linea impresa de asteristos para que sea visible en el programa. A continuación, ejecuta la función original utilizando la sintaxis result = func(*args, **kwargs). Finalmente, imprime el resultado devuelto por la función original y devuelve este resultado utilizando la declaración return result.

Para el **Punto 4**. La clase **join_data** toma un objeto self y tiene como objetivo combinar dos archivos CSV en un solo marco de datos.
El método comienza llamando al método **get_file_csv()** del objeto self, que devuelve una lista de los nombres de archivo de los archivos CSV que se van a unir. Luego, se crea una lista vacía BD que se llenará con los marcos de datos de cada archivo CSV.
A continuación, el método itera a través de la lista de nombres de archivo y usa la biblioteca pandas para leer cada archivo CSV y convertirlo en un marco de datos de pandas. Luego, agrega cada marco de datos a la lista BD.
Luego, el método usa conjuntos de Python para encontrar las columnas comunes entre los dos marcos de datos en la lista BD. La variable on_dat contiene la intersección de las columnas de ambos marcos de datos, que se utiliza como la columna de combinación en la unión.
El método imprime la intersección de las columnas y la primera columna de on_dat, que debería ser la columna de combinación. Finalmente, el método utiliza la función pd.merge() de pandas para combinar los dos marcos de datos utilizando el método de unión interna y la columna de combinación.
El resultado de la función join_data() es un solo marco de datos que contiene las columnas de ambos archivos CSV y solo las filas que tienen valores en común en la columna de combinación.

Sigo con las clases que muestra y permite selecionar algunas columnas por el usuario y luego las combina en un unico DataFrame que con la siguiente clase se imprime y se muestra al usuario.
Y paso a la clase que me sirve para el **Punto 5** que Normaliza los valores en las columnas que contienen cadenas de texto.
Este método recorre todas las columnas del conjunto de datos que se pasa como argumento y para aquellas que contienen valores de tipo cadena (str), 
permite al usuario fusionar algunos de los valores únicos en la columna en uno solo.

Empecé con el **Punto 14** el cual consta de  que resuelve la ecuación utilizando la biblioteca SymPy y traza la función resultante utilizando Matplotlib.
Resuelve ecuaciones y muestra una gráfica de la solución.
El método toma dos argumentos: eq, que es la ecuación a resolver como una cadena de texto, y n, que es el rango de valores en el eje X que se quiere mostrar en la gráfica. Si no puede graficarlo imprime una excepcion con el mensaje de Errorsiiiiiitoooo.

Pasé a crear el **Punto 8** y como es que la clase debe tener un método que reciba condiciones para los datos tipo cadena y filtre
los datos de la tabla con base en esa condición, utilicé 3 condiciones, que busca los valores : 
1. Exactos
2. Contenga
3. Inice 

Empieza el codigo del **Punto 12** que trata simplemente del Metodo para realizar la regresión lineal simple entre dos variables de las columnas numéricas en el archivo CSV y retornar los coeficientes de regresión lineal
se utiliza la clase LinearRegression de scikit-learn para realizar la regresión lineal.

Añado al **Punto 16** otro *decorador* que es el plot_graph y es utilizado para trazar una línea de regresión lineal simple en las columnas seleccionadas utilizando el método regreLinealSimple de otra clase CsvReader.

Continuo con el codigo que me ayuda a empezar a correr el programa y que empiece la diversion con el Usuario! En clase se expondrá y se hará un ejemplo de uso en vivo con el profesor y los compañeros para comprobar su uso :D 

Para finalizar la entrega y el **Punto 18** se documentan TODAS las funciones y métodos

En resumen y en general, estos métodos proporcionan una forma fácil y flexible de seleccionar y procesar datos en un dataframe de pandas. 
#La biblioteca SymPy y Matplotlib se utilizan para realizar operaciones matemáticas y visualizar los resultados de manera efectiva.

## XOXO 
### Con amor Daian




