import ast

import sys
sys.path.append('.')
from gpt_manager import GPTBasicManager

# Me sitúo en la misma carpeta que el script
# para poder leer los ficheros de prueba
import os
directorio_del_script = os.path.dirname(os.path.abspath(__file__))
os.chdir(directorio_del_script)

gpt = GPTBasicManager(model="gpt-3.5-turbo")

def test_loteunico():
    with open('query-4ESO-17-loteunico.txt','r',encoding='iso-8859-1') as query_file:
        queries = query_file.read()
        queries = ast.literal_eval(queries)
    gpt.initialize()
    responses, void = gpt.send_queries(query_id="",query_list=queries)
    print(responses)

def test_unoenuno():
    with open('query-4ESO-17-deunaenuna.txt','r',encoding='iso-8859-1') as query_file:
        queries = query_file.read()
        queries = ast.literal_eval(queries) 
    gpt.initialize()
    responses, _ = gpt.send_queries(query_id="",query_list=queries)
    print(responses)

test_loteunico()
test_unoenuno()

pass