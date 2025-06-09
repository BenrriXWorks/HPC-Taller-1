### One-liners para ejecutar desde Powershell :

1. Compilar:
```ps1
wsl -e bash -c "cd RUTA && source venv/bin/activate && python setup.py build_ext --inplace"
```
2. Ejecutar prueba:
```ps1
wsl -e bash -c "cd RUTA && source venv/bin/activate && python test_basic.py"
```
3. Ejecutar indice invertido:
```ps1
wsl -e bash -c "cd RUTA && source venv/bin/activate && python algorithm-to-improve/test_basic.py" 
```
^ Notar que si se coloca como parametro compare ejecuta con numpy y vergesort.

¡Recordar reemplazar la ruta por la ruta actual del proyecto!