#PARTE 1
#permite el procesamiento de lenguaje natural
import difflib
import nltk
#permite minimizar las palabras
from nltk.stem import SnowballStemmer#intanciamos el minimizador
stemmer=SnowballStemmer('spanish')
#permite convertir casi lo que sea en arreglos
import numpy
#herramienta de deep learning
import tflearn
import tensorflow
#from tensorflow.python.framework import ops
#permite crear bases de datos
import json
#permite crear numeros aleatorios
import random
#permite guardar los modelos de entramientos (mejora la velocidad, ya que no hay que entrenar desde 0)
import pickle
#permite descargar el paquete punkt
nltk.download('punkt')

print("finalizado")

import dload
#dload.git_clone("https://github.com/boomcrash/data_bot.git")

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path=dir_path.replace("\\","/")
with open("data_bot/data.json", 'r', encoding='utf-8') as file:
  database=json.load(file)

#funcion para cambiar por sinonimos.
import spacy
# Cargar el modelo de lenguaje en español de spaCy
from nltk.corpus import wordnet

nltk.download('wordnet')

nlp = spacy.load("es_core_news_sm")

def get_synonyms(token):
    synonyms = set()
    for synset in wordnet.synsets(token.text, lang='spa'):
        for lemma in synset.lemmas(lang='spa'):
            synonyms.add(lemma.name())
    return list(synonyms)

# Registra la extensión `synonyms` en el objeto `Token`
spacy.tokens.Token.set_extension("synonyms", getter=get_synonyms)

def rephrase_sentence(sentence):
    doc = nlp(sentence)
    rephrased_sentence = []
    for token in doc:
        if token.pos_ in ["ADJ", "NOUN", "VERB"]:
            synonyms = [syn for syn in token._.synonyms]
            if synonyms:
                rephrased_sentence.append(synonyms[0])
            else:
                rephrased_sentence.append(token.text)
        else:
            rephrased_sentence.append(token.text)
    return " ".join(rephrased_sentence)

#quitar tildes
def quitarTildes(frase):
  replacements = (
    ("á", "a"),
    ("é", "e"),
    ("í", "i"),
    ("ó", "o"),
    ("ú", "u"),
  )
  for a, b in replacements:
    frase = frase.replace(a, b).replace(a.upper(), b.upper())
  return frase

#PARTE 2  
words=[]
all_words=[]
tags= []
aux= []
auxA= []
auxB= []
training=[]
exit=[]
 
try:
  with open("Entrenamiento/brain.pickle","rb") as pickleBrain:
    all_words,tags,training,exit=pickle.load(pickleBrain)
except:
  for intent in database['intents']:
    #recorremos los tags
    for pattern in intent['patterns']:
      #separamos una frase en palabras
      auxWords=nltk.word_tokenize(pattern)
      #guardamos las palabras
      auxA.append(auxWords)
      auxB.extend(auxWords)
      #guardamos los tags
      aux.append(intent['tag'])
  #simbolos a ignorar
  ignore_words=['?','!','.',',','¿',"'","$","·","-","_","&","%","/","(",")", "=","#"]
  for w in auxB:
    if w not in ignore_words:
      words.append(w)
  #truco para evitar repetidos
  words=sorted(set(words))
  print(words)    
  tags=sorted(set(aux))
  print(tags)
  #convertimos a minuscula
  all_words=[stemmer.stem(w.lower()) for w in words]
  print("SIN ORDENAR")
  print(len(all_words))
  #ordenamos la lista
  ##print("ORDENADO")
  all_words=sorted(list(set(all_words)))
  ##print(all_words)
  #ordenamos los tags
  tags=sorted(tags)
  ##print("TAGS ORDENADO")
  ##print(tags)
  training=[]
  exit=[]
  #creamos una salida falsa
  null_exit=[0 for _ in range(len(tags))]
  #recorremos el auxiliar de palabras
  for i, document in enumerate(auxA):
    bucket=[]
    #hacemos minuscula y quitamos signos
    auxWords=[stemmer.stem(w.lower())for w in document if w!="?"]
    #recorremos las palabras
    for w in all_words:
      if w in auxWords:
        bucket.append(1)
      else:
        bucket.append(0)
    exit_row=null_exit[:]
    exit_row[tags.index(aux[i])]=1
    training.append(bucket)
    exit.append(exit_row)
    ##print(training)
    ##print(exit)
     #pasamos la lista del entrenamiento a un arreglo numpy
  training=numpy.array(training)
  ##print(training)
  exit=numpy.array(exit)
  ##print(exit)


  #crear un archivo pickle para almacenar los datos entrenados
  with open("Entrenamiento/brain.pickle","wb") as pickleBrain:
    pickle.dump((all_words,tags,training,exit),pickleBrain)

#print(training[0])



#PARTE 3

print(exit.shape)
print(training.shape)
# Crea el grafo de TensorFlow
graph = tensorflow.Graph()
with graph.as_default():

#creamos una red neuronal #pasamos longitud de la fila
    #creamos las neuronas conectadas

    #neuronas de entrada(input layers)
    net=tflearn.input_data(shape=[None,len(training[0])])


    #neuronas intermedias(hidden layers)
    net=tflearn.fully_connected(net,len(all_words),activation='tanh')
    net=tflearn.fully_connected(net,len(all_words),activation='relu')
    net=tflearn.dropout(net,0.9)

    
    #normalizar
    net = tflearn.batch_normalization(net)
    # Agregar una capa completamente conectada con 58 neuronas y activación 'softmax'
    net = tflearn.fully_connected(net, len(exit[0]), activation='softmax')

    # Definir la función de pérdida y optimizador
    net = tflearn.regression(net, optimizer='adam',learning_rate=0.001, loss='categorical_crossentropy')

    # Crear el modelo
    model = tflearn.DNN(net)

    # Después de crear el modelo
    #from keras.callbacks import EarlyStopping
    # early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    if os.path.isfile(dir_path+"/Entrenamiento/model.tflearn.index"):
      model.load(dir_path+"/Entrenamiento/model.tflearn")
    else:
      # Crear una instancia de EarlyStopCallback con un límite de 5 épocas sin mejora en la pérdida de validación
      import callbackEarlyStop
      early_stop = callbackEarlyStop.EarlyStopCallback(50)

      # Entrenar el modelo con los datos de entrada y salida
      model.fit(training, exit, validation_set=0.05, n_epoch=150, batch_size=20, show_metric=True)
      # Guardar el modelo
      model.save("Entrenamiento/model.tflearn")
    #print(len(all_words))
    #moldeamos el modelo (n_epoch son las veces que se revisa la base de datos para obtener respuestas)
    #bath_size son el # de entradas
    #show metric muestrav el contenido de la metrica
    #EVALUACION DEL MODELO
    print("EVALUACION DEL MODELO",model.evaluate(training,exit),"%")

def similar(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

def response(texto):
  # ojoooooooooooooooooooooo
  texto=quitarTildes(texto)
  if texto=="duerme":
    print("HA SIDO UN GUSTO, VUELVE PRONTO....")
    return False
  else:
    bucket=[0 for _ in range(len(all_words))]
    processed_sentence=nltk.word_tokenize(texto)
    processed_sentence=[stemmer.stem(palabra.lower()) for palabra in processed_sentence]
    for individual_word in processed_sentence:
      for i,palabra in enumerate(all_words):
        if palabra==individual_word:
          bucket[i]=1
    results=model.predict([numpy.array(bucket)])
    index_results=numpy.argmax(results)
    max=results[0][index_results]
    if max>0:
      #print(index_results)
      #print(max)
      target=tags[index_results]
      #print(tags)
      #print(target)
      for tagAux in database['intents']:
        if tagAux['tag']==target:
           lista_patrones=tagAux['patterns']
           entrada_usuario = texto # aquí asumimos que has capturado la entrada del usuario en una variable llamada "input_usuario"
           mejor_coincide = 0
           mejor_patron = ''
           for patron in lista_patrones:
               similitud = similar(patron, entrada_usuario) # aquí asumimos que tienes una función para medir la similitud entre dos cadenas de texto llamada "similar"
               if similitud > mejor_coincide:
                   mejor_coincide = similitud
                   mejor_patron = patron
           print("patron-coincide:", mejor_patron)
           
           #indice del patron
           indice=lista_patrones.index(mejor_patron)
           answer=tagAux['responses']
           print(answer)
           answer=answer[indice]
           print(answer)
      print("pregunta: ",texto,"\n respuesta: ",answer,"\n porcentaje: ",max,"\n tag: ",target)
    else:
      answer= "Lo siento, no tengo información actualizada sobre esta pregunta. Por favor, contáctenos a través de nuestro número de WhatsApp que se encuentra en nuestra página web para obtener la información actualizada."
      print("pregunta: ", texto,"\n respuesta: ",answer, "\n porcentaje: ", max)
    #rephrase_sentence(answer)
    return answer

print("HABLA CONMIGO")
bool=True

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

#NUEVO
@app.route("/bot", methods=['POST'])
def sms_reply():
  try:
    mensaje = request.json
    texto=mensaje["sms"]
    respuesta = str(response(str(texto)))
    return respuesta
  except Exception as e:
    print("error:",e)
    return str("puedes formular diferente la pregunta?")

@app.route("/products", methods=['GET'])
def getProducts():
  return "hola"


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)




