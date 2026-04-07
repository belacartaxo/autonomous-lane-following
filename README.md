# autonomous-lane-following

Baixar o weboots https://cyberbotics.com/doc/guide/installation-procedure


Abrir o worlds/city_default no weboots -> file > open world


Instalar dependencias
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt


Adicione o caminho do webots na ide, se for windows e estiver usando o vscode vai tem que adicionar algo do tipo no json das configurações de usuario

"python.analysis.extraPaths": [
    "C:/Program Files/Webots/lib/controller/python"
],
"python.autoComplete.extraPaths": [
    "C:/Program Files/Webots/lib/controller/python"
]