# TODO: reconvertir como unittest
import ast

from gpt_manager import GPTMockManager

# Me sitúo en la misma carpeta que el script
# para poder leer los ficheros de prueba
import os
directorio_del_script = os.path.dirname(os.path.abspath(__file__))
os.chdir(directorio_del_script)

gpt = GPTMockManager()

def test_loteunico():
    with open('query-4ESO-17-loteunico.txt','r',encoding='iso-8859-1') as query_file:
        queries = query_file.read()
        queries = ast.literal_eval(queries)
    gpt.initialize()
    responses = gpt.send_queries(query_id="",query_list=queries)
    print(responses)

def test_unoenuno():
    with open('query-4ESO-17-deunaenuna.txt','r',encoding='iso-8859-1') as query_file:
        queries = query_file.read()
        queries = ast.literal_eval(queries) 
    gpt.initialize()
    responses = gpt.send_queries(query_id="",query_list=queries)
    print(responses)

pass