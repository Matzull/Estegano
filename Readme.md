# Esteganografía con OneAPI

Se propone una solucion para el concurso de optimizacion de un programa de esteganografia mediante el uso del framework OneApi.

## Requisitos

Antes de compilar y ejecutar la aplicación, asegúrese de tener los siguientes requisitos:

- Un sistema operativo compatible con OneAPI, como Linux o Windows.
- El toolkit base de OneApi descargable desde:  [OneApi download](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit).
- Un compilador compatible con OneAPI, como `icpx` o `dpcpp`.
- Una cuenta en Intel DevCloud: [Intel® DevCloud for oneAPI](https://devcloud.intel.com/oneapi/).

## Compilación

Para compilar la aplicación, ejecute el siguiente comando en la línea de comandos:

```bash
user@host:~/ $ make
```

Este comando compilará la aplicación utilizando el compilador de OneAPI y las opciones de optimización recomendadas. Los archivos objeto se eliminarán automáticamente después de la compilación.

## Ejecución

El ejecutable final y los archivos necesarios para su funcionamiento se encuentran en el directorio `/builds` y `/imgs` respectivamente. Los archivos de salida del programa se guardan en el directorio `/imgs/Out`.
</br>
</br>
Si se desea ejecutar el codigo en un nodo con gpu de un jupyter notebook se debera utilizar el comando:

```bash
user@host:~/ $ qsub -I -l nodes=1:gpu:ppn=2 -d .
```

Una vez compilado, puede ejecutar la aplicación utilizando el siguiente comando:

```bash
user@host:~/ $ ./run.sh
```

Este comando ejecutará el archivo de script `run.sh`, que iniciará la aplicación con la configuración recomendada. Si desea cambiar la configuración, edite el archivo de script antes de ejecutarlo.

</br>
Se detallan los parametros necesarios para ejecutar la aplicacion manualmente:
</br>
Es necesario establecer las variables de entorno mediante el comando.

```bash
user@host:~/ $ source /opt/intel/oneapi/setvars.sh
```

*`imagen_entrada.png` `logo.png` `image_salida.png`*

- `imagen_entrada.png`: Imagen de entrada donde se va a ocultar el mensaje.
- `logo.png`: Imagen con el logo o mensaje a ocultar.
- `image_salida.png`: Imagen de salida correspondiente a la imagen de entrada con el mensaje oculto.

Además genera un fichero logo_out.png que corresponde al mensaje "descifrado" o recuperado.
