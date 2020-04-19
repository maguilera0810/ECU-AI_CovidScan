Descripción del Servidor

1) Requerimientos
    Librerias:
        - Flask
        - SQLalchemy
        - Tensorflow 
        - Numpy
        - OpenCV

    Adicionales:
        - Arquitectura del modelo entrenado y sus respectivos pesos para realizar la tarea de predicción.
        - Base de Datos basada en PostgreSQL, creada en la página https://www.heroku.com

2) Funcionamiento
    En primer lugar, para el correcto funcionamiento del servidor, se realizan 
    los siguientes pasos:
        - Establecer la dirección de la base de datos, que se encuentra en las credenciales al ingresar a https://www.heroku.com
        - Establecer la dirección en la cuál se van a almacenar las imágenes subidas por los distintos usuarios
        - Cargar la arquitectura del modelo (Formato Json). Luego de esto, cargar los respectivos pesos a dicho modelo.

    Cada una de las distintas rutas tiene un objetivo en específico:
        - @app.route('/', methods=['GET']) : Siendo la ruta principal, es lo primero que se muestra en la pantalla. Aquí, el 
        usuario va a poder llenar un formulario en el que se consultan sus datos personales y temas relacionados a la salud
        del mismo. De igual manera, se va a poder subir una imágen al servidor, la cual va a ser procesada para realizar 
        la predicción. 

        -@app.route('/uploader', methods=['GET', 'POST']): Una vez llenados los campos en la ruta anterior, se procede a navegar
        a esta página. En este punto, se mostrará un resumen con la información del usuario del cuál se está realizando la predicción.
        Además, se mostrará su respectiva imágen con los porcentajes que obtuvo de cada opción. Por ejemplo, el diagnóstico final fue:

                                        DIAGNOSTICS = {
                                        'covid-19': 0.90,
                                        'Pneumonia': 0.8,
                                        'Normal image': 0.2

        -@app.route('/listado', methods=['GET', 'POST']): En esta ruta, se mostrarán un listado con todos los usuarios que han  usado esta
        plataforma, cada uno con un respectivo resumén de toda la información subida.. 
}
3) Link Youtube
	https://youtu.be/DgBVt51vTpg
